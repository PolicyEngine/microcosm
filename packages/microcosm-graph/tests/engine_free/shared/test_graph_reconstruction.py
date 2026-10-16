"""Declaration-only schemas rebuilt for builds that recorded no run binding."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess

import pytest

from microcosm.graph import graph_schema, graph_to_json
from microcosm.graph.canonical import canonical_json
from microcosm.graph.orrery import orrery_document_from_schema, orrery_json_from_schema
from microcosm.graph.reconstruction import (
    DECLARATION_ONLY,
    RECONSTRUCTION_EXTENSION,
    RECONSTRUCTION_PROTOCOL,
    reconstructed_schema,
    reconstruction_marker,
    validate_reconstruction,
)
from microcosm.graph.schema import validate_graph_schema
from test_support.microcosm_graph.schema import (
    compiled_default_population_graph,
    compiled_graph,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-graph")
_RECONSTRUCTION_VERIFY = (
    _TEST_PATHS.repository / "tools" / "orrery-contract" / "verify-reconstruction.mjs"
)
RECORDED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def recorded_declaration() -> bytes:
    """The bytes a build wrote to graph.json for its declared graph."""

    return graph_to_json(compiled_graph().graph).encode("utf-8")


def input_record(name: str, role: str, payload: bytes, verification: str) -> dict:
    return {
        "name": name,
        "role": role,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "verification": verification,
    }


def marker_for(declaration: bytes, **revision_overrides) -> dict:
    revision = {
        "git_commit": RECORDED_COMMIT,
        "git_dirty": False,
        "revision_origin": "recorded",
        **revision_overrides,
    }
    return reconstruction_marker(
        **revision,
        inputs=[
            input_record(
                "rowwise_candidate_manifest.json",
                "code_pin",
                b'{"git_commit":"recorded"}',
                "recorded_digest_matched",
            ),
            input_record(
                "graph.json", "declaration", declaration, "no_recorded_digest"
            ),
        ],
    )


def test_reconstructed_schema_carries_the_marker_into_orrery_metadata():
    declaration = recorded_declaration()
    marker = marker_for(declaration)

    schema = reconstructed_schema(declaration, marker)

    assert schema["extensions"] == {RECONSTRUCTION_EXTENSION: marker}
    assert schema["graph_sha256"] == hashlib.sha256(declaration).hexdigest()
    assert marker["protocol"] == RECONSTRUCTION_PROTOCOL
    assert marker["reconstructed"] is True
    assert marker["basis"] == DECLARATION_ONLY
    assert marker["source_revision"] == {
        "git_commit": RECORDED_COMMIT,
        "git_dirty": False,
        "origin": "recorded",
    }
    # Inputs are listed once each, in name order, whatever order they came in.
    assert [item["name"] for item in marker["inputs"]] == [
        "graph.json",
        "rowwise_candidate_manifest.json",
    ]
    assert marker["scope"].strip()
    document = orrery_document_from_schema(schema)
    carried = document["metadata"]["microcosm"]["extensions"][RECONSTRUCTION_EXTENSION]
    assert carried == marker


def test_reconstructed_document_claims_no_execution():
    declaration = recorded_declaration()

    document = orrery_document_from_schema(
        reconstructed_schema(declaration, marker_for(declaration))
    )

    assert not {"activities", "artifacts", "receipts"} & document.keys()
    assert all("statuses" not in node for node in document["nodes"])
    assert "execution" not in document["metadata"]
    assert sum(node["kind"] == "operation" for node in document["nodes"]) == len(
        compiled_graph().graph.nodes
    )


def test_declaration_that_does_not_reproduce_its_recorded_bytes_is_refused():
    # The same graph, re-indented: it parses, but the current serializer would
    # not write these bytes, so the projection would not be the recorded file.
    reformatted = json.dumps(json.loads(recorded_declaration()), indent=2).encode()

    with pytest.raises(ValueError, match="does not reproduce its recorded bytes"):
        reconstructed_schema(reformatted, marker_for(reformatted))


def test_marker_must_name_the_declaration_it_was_built_from():
    other_declaration = graph_to_json(compiled_default_population_graph().graph).encode(
        "utf-8"
    )

    with pytest.raises(ValueError, match="does not name this declaration"):
        reconstructed_schema(other_declaration, marker_for(recorded_declaration()))


def _short_commit(marker: dict) -> None:
    marker["source_revision"]["git_commit"] = "0123456"


def _uppercase_commit(marker: dict) -> None:
    marker["source_revision"]["git_commit"] = RECORDED_COMMIT.upper()


def _guessed_origin(marker: dict) -> None:
    marker["source_revision"]["origin"] = "guessed"


def _textual_dirty_flag(marker: dict) -> None:
    marker["source_revision"]["git_dirty"] = "no"


def _no_declaration_input(marker: dict) -> None:
    for item in marker["inputs"]:
        item["role"] = "code_pin"


def _two_declaration_inputs(marker: dict) -> None:
    for item in marker["inputs"]:
        item["role"] = "declaration"


def _duplicate_input_names(marker: dict) -> None:
    marker["inputs"][1]["name"] = marker["inputs"][0]["name"]


def _unsorted_inputs(marker: dict) -> None:
    marker["inputs"].reverse()


def _nested_input_path(marker: dict) -> None:
    marker["inputs"][1]["name"] = "../rowwise_candidate_manifest.json"


def _unknown_verification(marker: dict) -> None:
    marker["inputs"][1]["verification"] = "trusted"


def _negative_input_size(marker: dict) -> None:
    marker["inputs"][1]["bytes"] = -1


def _unknown_marker_field(marker: dict) -> None:
    marker["recorded_execution"] = True


def _recorded_basis(marker: dict) -> None:
    marker["basis"] = "recorded_execution"


def _not_reconstructed(marker: dict) -> None:
    marker["reconstructed"] = False


def _other_protocol(marker: dict) -> None:
    marker["protocol"] = "microcosm.graph.reconstruction.v0"


@pytest.mark.parametrize(
    "corrupt",
    [
        _short_commit,
        _uppercase_commit,
        _guessed_origin,
        _textual_dirty_flag,
        _no_declaration_input,
        _two_declaration_inputs,
        _duplicate_input_names,
        _unsorted_inputs,
        _nested_input_path,
        _unknown_verification,
        _negative_input_size,
        _unknown_marker_field,
        _recorded_basis,
        _not_reconstructed,
        _other_protocol,
    ],
    ids=lambda corrupt: corrupt.__name__.strip("_"),
)
def test_marker_rejects_malformed_provenance(corrupt):
    declaration = recorded_declaration()
    marker = copy.deepcopy(marker_for(declaration))
    corrupt(marker)

    with pytest.raises(ValueError, match="Graph reconstruction"):
        reconstructed_schema(declaration, marker)


def test_saved_reconstructed_schema_keeps_its_marker_when_reprojected():
    declaration = recorded_declaration()
    marker = marker_for(declaration, git_dirty=None, revision_origin="caller_asserted")
    schema = reconstructed_schema(declaration, marker)

    # A later projection (a collector re-projecting stored inputs, say) reads
    # only the saved graph.schema.json bytes; the marker travels inside them.
    restored = validate_graph_schema(json.loads(canonical_json(schema)))

    assert validate_reconstruction(restored) == marker
    assert orrery_json_from_schema(restored) == orrery_json_from_schema(schema)
    assert validate_reconstruction(graph_schema(compiled_graph())) is None
    relabelled = graph_schema(
        compiled_default_population_graph(),
        extensions={RECONSTRUCTION_EXTENSION: marker},
    )
    with pytest.raises(ValueError, match="does not name this declaration"):
        validate_reconstruction(relabelled)


def test_public_orrery_parser_accepts_reconstructed_document():
    declaration = recorded_declaration()
    payload = orrery_json_from_schema(
        reconstructed_schema(declaration, marker_for(declaration)),
        title="INVENTED · reconstructed declaration",
    )

    result = subprocess.run(
        ["node", str(_RECONSTRUCTION_VERIFY)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "Orrery accepted the reconstructed declaration.\n"
