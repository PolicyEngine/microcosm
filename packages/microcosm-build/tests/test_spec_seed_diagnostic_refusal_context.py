"""Source-only context formatting: no audit install or denied file operation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "spec_seed_diagnostic_context_test",
        root / "tools/spec_seed_identity_diagnostics.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "scope", "shown"),
    (
        ("/repo/tools/example.py", "repository", "tools/example.py"),
        (
            "/repo/packages/example/schema.json",
            "repository",
            "packages/example/schema.json",
        ),
        ("/repo/data/invented.csv", "repository", None),
        ("/repo/.env", "repository", None),
        ("/repo/credentials.json", "repository", None),
        ("/repo/example.err.log", "repository", None),
        ("/unlisted/identity/example.py", "unlisted", "example.py"),
        ("/opt/python/lib/example.py", "system_opt", "python/lib/example.py"),
    ),
)
def test_path_classification_is_bounded_and_redacts_data_and_credentials(
    diagnostic, value, scope, shown
):
    result = diagnostic.refusal_path_context(
        value, (("repository", Path("/repo")), ("system_opt", Path("/opt")))
    )
    assert result["scope"] == scope
    assert result["path"] == shown
    assert result["name_redacted"] is (shown is None)


def test_requested_and_resolved_code_scopes_remain_distinct_without_io(diagnostic):
    # Literal examples only: no symlink or path is created, resolved or opened.
    payload = diagnostic.read_refusal_context(
        "READ_SCOPE",
        "open",
        Path("/opt/python/lib/loader.py"),
        "/environment/lib/loader.py",
        (("environment", Path("/environment")), ("system_opt", Path("/opt"))),
    )
    value = json.loads(payload)
    assert value["requested"]["scope"] == "environment"
    assert value["resolved"]["scope"] == "system_opt"
    assert value["paths_are_metadata_only"] is True
    assert len(payload) <= 4096 and len(value["frames"]) <= 8


def test_context_never_serializes_locals_or_exception_text(diagnostic):
    private_local = "invented-secret-value-that-must-never-be-exported"
    error = RuntimeError(private_local)
    payload = diagnostic.read_refusal_context(
        "READ_SCOPE",
        "open",
        Path("/repo/credentials.json"),
        "/repo/credentials.json",
        (("repository", Path("/repo")),),
    )
    assert str(error).encode() not in payload
    value = json.loads(payload)
    assert value["requested"]["path"] is None
    assert value["resolved"]["path"] is None
    assert value["code"] == "READ_SCOPE" and value["event"] == "open"
    assert all(set(frame) == {"file", "function", "line"} for frame in value["frames"])
    assert len(payload) <= 4096


@pytest.mark.parametrize(
    ("path", "label"),
    (
        ("/etc/os-release", "etc_os_release"),
        ("/usr/lib/os-release", "usr_lib_os_release"),
        ("/lib/os-release", "lib_os_release"),
        ("/etc/localtime", "etc_localtime"),
        (
            f"/runtime/lib/python{sys.version_info.major}{sys.version_info.minor}.zip",
            "python_stdlib_zip_candidate",
        ),
        ("/runtime/lib/invented-data.zip", None),
    ),
)
def test_fixed_os_metadata_labels_do_not_expose_data_path_text(diagnostic, path, label):
    value = diagnostic.refusal_path_context(path, (("runtime", Path("/runtime")),))
    assert value["known_path_label"] == label
    assert value["path"] is None and value["name_redacted"] is True


def test_terminal_refusal_retains_own_context_after_an_earlier_probe(diagnostic):
    first = diagnostic.encoded({"code": "DATA_FILE", "event": "open"})
    last = diagnostic.read_refusal_context(
        "READ_SCOPE",
        "open",
        Path("/usr/lib/os-release"),
        "/etc/os-release",
        (("system_usr", Path("/usr")), ("system_etc", Path("/etc"))),
    )
    swallowed = diagnostic.RefusalError("DATA_FILE", boundary_context=first)
    terminal = diagnostic.RefusalError("READ_SCOPE", boundary_context=last)
    assert str(swallowed) == "DATA_FILE" and str(terminal) == "READ_SCOPE"
    assert type(terminal.boundary_context) is bytes
    assert terminal.boundary_context == last and swallowed.boundary_context == first
    context = json.loads(terminal.boundary_context)
    assert context["requested"]["known_path_label"] == "etc_os_release"
    assert context["resolved"]["known_path_label"] == "usr_lib_os_release"
    assert context["code"] != json.loads(swallowed.boundary_context)["code"]
    assert diagnostic.RefusalError("WALL").boundary_context is None


@pytest.mark.parametrize("value", (bytearray(b"{}"), b"x" * 4097))
def test_exception_context_requires_immutable_bounded_bytes(diagnostic, value):
    with pytest.raises(TypeError, match="REFUSAL_CONTEXT"):
        diagnostic.RefusalError("READ_SCOPE", boundary_context=value)
