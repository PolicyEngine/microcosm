"""Canonical JSON and normative-declaration identity contracts."""

from __future__ import annotations

import hashlib
from types import MappingProxyType

import pytest

from microcosm.graph.canonical import canonical_json, normative, sha256_domain
from microcosm.graph.decl import Node, Owned, Ownership, SourceRef


def test_canonical_json_has_closed_deterministic_encoding() -> None:
    value = MappingProxyType(
        {
            "z": 2,
            "a": (-0.0, 1.0, True, None, "café", Ownership.ABSENT),
        }
    )
    assert canonical_json(value) == (
        b'{"a":[-0.0,1.0,true,null,"caf\xc3\xa9","absent"],"z":2}'
    )


def test_canonical_json_rejects_non_json_values_and_keys() -> None:
    with pytest.raises(TypeError, match="not part"):
        canonical_json({"bad": object()})
    with pytest.raises(TypeError, match="string keys"):
        canonical_json({1: "bad"})
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(float("nan"))
    with pytest.raises(TypeError, match="not part"):
        canonical_json(SourceRef("survey", "csv-tables"))


def test_domain_hash_prefixes_the_domain_and_nul() -> None:
    payload = b"payload"
    assert (
        sha256_domain("node", payload) == hashlib.sha256(b"node\0payload").hexdigest()
    )
    assert sha256_domain("node", payload) != sha256_domain("frame", payload)
    with pytest.raises(ValueError, match="must not contain NUL"):
        sha256_domain("node\0nested", payload)


def test_normative_strips_only_declaration_descriptive_fields() -> None:
    node = Node(
        "fill",
        "toy.fill@1",
        outputs=(Owned("person", "x", "float64"),),
        params={"description": "this parameter changes behavior"},
        description="human explanation",
        citation="paper",
    )
    projection = normative(node)
    assert isinstance(projection, dict)
    assert "description" not in projection
    assert "citation" not in projection
    assert projection["params"] == {"description": "this parameter changes behavior"}

    assert normative(SourceRef("survey", "csv-tables", "human words")) == {
        "name": "survey",
        "codec": "csv-tables",
    }
