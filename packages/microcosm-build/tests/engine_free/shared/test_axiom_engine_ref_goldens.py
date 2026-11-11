"""Absent views leave every Axiom engine reference byte for byte unchanged.

The golden holds references generated at the base of S2b PR 1 (main
``4a36edcf``) for the six NZ binding rows frozen in test support,
``unreached_roots`` included, over a synthetic tree, and for the registry
and adapter relation fixtures
(:mod:`test_support.microcosm_build.axiom_ref_goldens`). A reference that
gains ``"views": []`` or ``"views": null`` without a declared view fails
here. The real-tree check runs only with ``POPULACE_RULESPEC_NZ``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import test_support.microcosm_build.axiom_ref_goldens as golden_helpers
from microcosm.build.transport.registry import build_transport_registry
from test_support.microcosm_build.axiom_ref_goldens import (
    GOLDEN,
    NZ_BINDINGS,
    engine_refs,
)


def test_engine_refs_match_the_base_golden_byte_for_byte(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    assert engine_refs(tmp_path / "refs") == golden
    for name, reference in golden.items():
        assert "views" not in json.loads(reference), name


def test_ci_golden_source_ignores_live_nz_bindings(tmp_path, monkeypatch):
    expected = golden_helpers.nz_document(tmp_path / "before")
    monkeypatch.setattr(
        golden_helpers, "NZ_BINDINGS", tmp_path / "missing-live-bindings.json"
    )
    assert golden_helpers.nz_document(tmp_path / "after") == expected


RULESPEC_NZ = os.environ.get("POPULACE_RULESPEC_NZ")
REAL_GOLDEN = GOLDEN.with_name("nz_real_axiom_engine_refs.json")


@pytest.mark.skipif(
    not RULESPEC_NZ,
    reason="needs a rulespec-nz export at the pinned commit at POPULACE_RULESPEC_NZ",
)
def test_nz_real_refs_match_their_frozen_copy(tmp_path):
    """NZ's six real references against a frozen copy (the tree, not the engine).

    This check reads NZ's live binding resource and its declared RuleSpec
    commit, unlike the synthetic CI golden. To regenerate intentionally,
    export that pinned tree, set ``POPULACE_RULESPEC_NZ`` to the export,
    remove ``REAL_GOLDEN``, and run this test. It writes the frozen file and
    fails; review the result, then rerun to verify it. Keep the existing copy
    when checking compatibility with a later source head.
    """
    import microcosm.build.transport.registry as registry_module
    from microcosm.frame import VariableMetadata
    from microcosm.frame.adapters.axiom import AxiomEngine
    from test_support.microcosm_build.transport_population import UNIT_RULE

    document = json.loads(NZ_BINDINGS.read_text())
    document["engine"] = {"commit": "a" * 40, "wheel_sha256": "b" * 64}
    entities = {
        variable: row["entity"]
        for row in document["bindings"]
        for variable in row["variables"]
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry_module, "assert_no_relations", lambda *args: None)
        patch.setattr(
            AxiomEngine,
            "variable_metadata",
            lambda engine, name: VariableMetadata(
                name, entities[name], "float", "year"
            ),
        )
        patch.setattr(AxiomEngine, "graph_dtype", lambda engine, name: "float64")
        registry = build_transport_registry(
            document, Path(RULESPEC_NZ), unit_rule=UNIT_RULE
        )
    refs = dict(sorted(registry.engine_refs.items()))
    assert all("views" not in json.loads(ref) for ref in refs.values())
    if not REAL_GOLDEN.exists():
        REAL_GOLDEN.write_text(json.dumps(refs, indent=1, sort_keys=True) + "\n")
        pytest.fail(f"Froze {REAL_GOLDEN.name}; commit it at the base and rerun.")
    assert refs == json.loads(REAL_GOLDEN.read_text())
