"""The Orrery projection runs without the build runtime.

A collector worker will project a saved compiler schema and execution overlay
file without the content store or the population libraries. These tests run
that projection in a fresh interpreter that refuses to import numpy, pandas,
microcosm-frame or the store, and compare its bytes with the projection a
build makes in-process.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

import microcosm.graph as graph_api
from microcosm.graph import (
    collect_execution_evidence,
    execution_overlay,
    graph_schema,
    record_run_binding,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.orrery import main, orrery_json_from_schema
from microcosm.graph.presentation import PRESENTATION_EXTENSION, PRESENTATION_PROTOCOL
from test_support.paths import paths_for

_PATHS = paths_for("microcosm-graph")
if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location("_toy", _PATHS.tests / "_toy.py")
    _TOY = importlib.util.module_from_spec(_SPEC)
    sys.modules["_toy"] = _TOY
    _SPEC.loader.exec_module(_TOY)
toy = sys.modules["_toy"]

#: Modules a projection process must be able to do without.
RUNTIME_MODULES = ("numpy", "pandas", "microcosm.frame", "microcosm.graph.store")

#: Installs the refusal before anything else imports, then runs the body.
_REFUSE_RUNTIME_IMPORTS = r"""
import importlib.abc, json, sys

refused = tuple(sys.argv[1].split(","))


def is_refused(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in refused)


class RefuseRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if is_refused(name):
            raise ImportError(f"the Orrery projection imported {name}")
        return None


sys.meta_path.insert(0, RefuseRuntimeImports())
"""

_RESTRICTED_PROJECTION = (
    _REFUSE_RUNTIME_IMPORTS
    + r"""
from pathlib import Path

from microcosm.graph.execution_overlay import load_execution_overlay
from microcosm.graph.orrery import orrery_json_from_schema

schema_path, overlay_path, output_path = (Path(argument) for argument in sys.argv[2:5])
schema = json.loads(schema_path.read_bytes())
overlay = load_execution_overlay(overlay_path)
output_path.write_text(
    orrery_json_from_schema(schema, execution=overlay), encoding="utf-8"
)
print(json.dumps(sorted(name for name in sys.modules if is_refused(name))))
"""
)

_RESTRICTED_COMMAND = (
    _REFUSE_RUNTIME_IMPORTS
    + r"""
import runpy

sys.argv = ["microcosm.graph.orrery", *sys.argv[2:]]
runpy.run_module("microcosm.graph.orrery", run_name="__main__")
"""
)


def _run_restricted(script: str, *arguments: str, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script, ",".join(RUNTIME_MODULES), *arguments],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=cwd,
    )


@pytest.fixture
def recorded(tmp_path):
    """A computing phase and a replay of the toy graph, with their schema."""
    computed = toy.run_toy(toy.full_graph(), tmp_path)
    replayed = toy.run_toy(
        computed.compiled.graph,
        tmp_path,
        sources=computed.sources,
        registry=computed.registry,
        store=computed.store,
    )
    schema = graph_schema(
        computed.compiled,
        extensions={
            PRESENTATION_EXTENSION: {
                "protocol": PRESENTATION_PROTOCOL,
                "scope": {"id": "fixture", "label": "Synthetic country"},
                "groups": [
                    {"id": "sources", "label": "Sources", "sources": ["survey"]},
                    {
                        "id": "work",
                        "label": "Processing",
                        "operations": list(computed.compiled.order),
                    },
                ],
            }
        },
    )
    runs = [
        record_run_binding(
            computed.compiled, computed.manifest, attempt_id="attempt", phase="first"
        ),
        record_run_binding(
            replayed.compiled, replayed.manifest, attempt_id="attempt", phase="replay"
        ),
    ]
    overlay = collect_execution_evidence(schema, runs=runs, store=computed.store)
    return schema, overlay


def test_orrery_projection_runs_where_numpy_pandas_frame_and_store_cannot_import(
    recorded, tmp_path
):
    schema, overlay = recorded
    schema_path = tmp_path / "graph.schema.json"
    overlay_path = tmp_path / "graph.execution.json"
    output_path = tmp_path / "graph.orrery.json"
    schema_path.write_bytes(canonical_json(schema))
    overlay_path.write_bytes(execution_overlay.execution_overlay_bytes(overlay))

    completed = _run_restricted(
        _RESTRICTED_PROJECTION,
        str(schema_path),
        str(overlay_path),
        str(output_path),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert json.loads(completed.stdout) == []
    assert output_path.read_bytes() == (
        orrery_json_from_schema(schema, execution=overlay).encode("utf-8")
    )


def test_orrery_command_runs_where_numpy_pandas_frame_and_store_cannot_import(
    recorded, tmp_path
):
    schema, _overlay = recorded
    schema_path = tmp_path / "graph.schema.json"
    schema_path.write_bytes(canonical_json(schema))
    restricted_output = tmp_path / "restricted" / "graph.orrery.json"
    in_process_output = tmp_path / "in-process" / "graph.orrery.json"

    completed = _run_restricted(
        _RESTRICTED_COMMAND,
        "--schema",
        str(schema_path),
        "--output",
        str(restricted_output),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert main(["--schema", str(schema_path), "--output", str(in_process_output)]) == 0
    assert restricted_output.read_bytes() == in_process_output.read_bytes()


def test_package_runtime_exports_still_resolve_from_the_package():
    from microcosm.graph import executor, store

    for name in graph_api.__all__:
        assert getattr(graph_api, name) is not None, name
    assert set(graph_api.__all__) <= set(dir(graph_api))
    assert graph_api.ContentStore is store.ContentStore
    assert graph_api.run_graph is executor.run_graph
    with pytest.raises(AttributeError, match="not_an_export"):
        graph_api.not_an_export  # noqa: B018


def test_execution_overlay_file_round_trips_to_byte_identical_orrery_json(
    recorded, tmp_path
):
    schema, overlay = recorded
    overlay_path = tmp_path / "graph.execution.json"
    overlay_path.write_bytes(execution_overlay.execution_overlay_bytes(overlay))

    loaded = execution_overlay.load_execution_overlay(overlay_path)

    assert overlay_path.read_bytes() == canonical_json(overlay)
    assert loaded["protocol"] == "microcosm.graph.execution-evidence.v1"
    assert loaded == overlay
    assert orrery_json_from_schema(schema, execution=loaded) == (
        orrery_json_from_schema(schema, execution=overlay)
    )


_OVERLAY_PROTOCOL_FIELD = b'"protocol":"microcosm.graph.execution-evidence.v1"'


@pytest.mark.parametrize(
    "payload,message",
    [
        (b'{"protocol":"microcosm.graph.execution-input.v1"}', "evidence.v1 object"),
        (b'["microcosm.graph.execution-evidence.v1"]', "evidence.v1 object"),
        (
            b"{" + _OVERLAY_PROTOCOL_FIELD + b',"history":"a","history":"b"}',
            "duplicate",
        ),
        (b"{" + _OVERLAY_PROTOCOL_FIELD + b',"history":NaN}', "finite JSON numbers"),
        (b"{" + _OVERLAY_PROTOCOL_FIELD + b',"history":1e400}', "finite numbers"),
        (b"{" + _OVERLAY_PROTOCOL_FIELD.replace(b"evidence", b"\xff") + b"}", "utf-8"),
    ],
    ids=[
        "another protocol",
        "not an object",
        "duplicate key",
        "NaN literal",
        "overflowing number",
        "not UTF-8",
    ],
)
def test_execution_overlay_reader_refuses_malformed_files(tmp_path, payload, message):
    overlay_path = tmp_path / "graph.execution.json"
    overlay_path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        execution_overlay.load_execution_overlay(overlay_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO")
def test_execution_overlay_reader_refuses_a_fifo_without_waiting_for_a_writer(
    tmp_path,
):
    overlay_path = tmp_path / "graph.execution.json"
    os.mkfifo(overlay_path)
    with pytest.raises(ValueError, match="regular file"):
        execution_overlay.load_execution_overlay(overlay_path)


def test_execution_overlay_reader_refuses_a_file_over_the_byte_limit(
    recorded, tmp_path, monkeypatch
):
    _schema, overlay = recorded
    overlay_path = tmp_path / "graph.execution.json"
    payload = execution_overlay.execution_overlay_bytes(overlay)
    overlay_path.write_bytes(payload)
    monkeypatch.setattr(execution_overlay, "_MAX_OVERLAY_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        execution_overlay.load_execution_overlay(overlay_path)


def test_execution_overlay_writer_refuses_a_value_without_the_overlay_protocol(
    recorded,
):
    _schema, overlay = recorded
    another_protocol = {**overlay, "protocol": "microcosm.graph.execution-input.v1"}
    for value in (another_protocol, [overlay]):
        with pytest.raises(ValueError, match="evidence.v1 object"):
            execution_overlay.execution_overlay_bytes(value)
