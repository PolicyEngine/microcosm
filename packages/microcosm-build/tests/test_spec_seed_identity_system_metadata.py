"""Invented metadata controls; no main, engine, data, child or real audit hook."""

import importlib.util
import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("system_metadata_controls", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def invented_uname(diagnostic, monkeypatch):
    observed = SimpleNamespace(
        sysname="Linux",
        nodename="invented-host",
        release="invented-release",
        version="invented-version",
        machine="invented_arch",
    )
    info = platform.uname_result(
        observed.sysname,
        observed.nodename,
        observed.release,
        observed.version,
        observed.machine,
    )
    monkeypatch.setattr(diagnostic.sys, "platform", "linux")
    monkeypatch.setattr(diagnostic.os, "uname", lambda: observed)
    monkeypatch.setattr(diagnostic.platform, "uname", lambda: info)

    def forbidden(*args, **kwargs):
        pytest.fail("processor metadata attempted a subprocess")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    return info, observed


def test_real_cached_property_uses_machine_without_replacing_stdlib(
    diagnostic, invented_uname
):
    info, observed = invented_uname
    before = (
        platform.uname,
        platform.processor,
        platform.uname_result,
        vars(type(info))["processor"],
    )
    bootstrap = {}
    state = diagnostic.prime_processor_metadata(bootstrap)
    assert platform.processor() == observed.machine
    assert tuple(info) == (
        observed.sysname,
        observed.nodename,
        observed.release,
        observed.version,
        observed.machine,
        observed.machine,
    )
    assert bootstrap["system_metadata"]["uname_p_parity_claimed"] is False
    assert bootstrap["system_metadata"]["source"] == "os.uname.machine"
    assert before == (
        platform.uname,
        platform.processor,
        platform.uname_result,
        vars(type(info))["processor"],
    )
    diagnostic.verify_processor_metadata(state, bootstrap)
    # An already-cached identical machine value is not silently altered.
    again = diagnostic.prime_processor_metadata(bootstrap)
    assert again == state


def test_conflicting_observed_processor_cache_is_not_overwritten(
    diagnostic, invented_uname
):
    info, _ = invented_uname
    info.processor = "different_observed_processor"
    bootstrap = {}
    with pytest.raises(diagnostic.RefusalError, match="^PROCESSOR_CACHE_CONFLICT$"):
        diagnostic.prime_processor_metadata(bootstrap)
    assert info.processor == "different_observed_processor"
    assert bootstrap == {}


@pytest.mark.parametrize("change", ("cached_value", "os_machine", "evidence"))
def test_metadata_change_refuses(diagnostic, invented_uname, change):
    info, observed = invented_uname
    bootstrap = {}
    state = diagnostic.prime_processor_metadata(bootstrap)
    if change == "cached_value":
        info.processor = "changed"
    elif change == "os_machine":
        observed.machine = "changed"
    else:
        bootstrap["system_metadata"]["uname_p_parity_claimed"] = True
    with pytest.raises(diagnostic.RefusalError, match="^PROCESSOR_METADATA_CHANGED$"):
        diagnostic.verify_processor_metadata(state, bootstrap)


@pytest.fixture
def audited(diagnostic, monkeypatch):
    hooks = []
    monkeypatch.setattr(diagnostic.sys, "addaudithook", hooks.append)
    monkeypatch.setattr(diagnostic.os, "getpid", lambda: 321)
    monkeypatch.setattr(diagnostic.os, "readlink", lambda _: "pipe:[invented]")
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/invented/repo")))

    def resolve(path):
        return Path("/proc/321/maps") if path == Path("/proc/self/maps") else path

    monkeypatch.setattr(Path, "resolve", resolve)
    refusals = diagnostic.install_boundary(
        Path("/invented/repo"), Path("/invented/owned"), Path("/invented/output")
    )
    assert len(hooks) == 1
    return hooks[0], refusals


@pytest.mark.parametrize("path", ("/proc/self/maps", "/proc/321/maps"))
def test_only_logical_and_resolved_own_maps_are_admitted(audited, path):
    hook, refusals = audited
    hook("open", (path, "r", os.O_RDONLY))
    assert refusals == []


@pytest.mark.parametrize(
    "kind", ("other_process", "other_file", "relative", "descriptor", "child")
)
def test_maps_alias_does_not_admit_other_paths_descriptors_or_children(
    diagnostic, audited, kind
):
    hook, refusals = audited
    path = {
        "other_process": "/proc/322/maps",
        "other_file": "/proc/321/mem",
        "relative": "maps",
        "descriptor": 55,
        "child": None,
    }[kind]
    code = "NETWORK_OR_CHILD" if kind == "child" else "READ_SCOPE"
    with pytest.raises(diagnostic.RefusalError, match="^" + code + "$"):
        if kind == "child":
            hook("subprocess.Popen", ("uname", ["uname", "-p"], None, None))
        else:
            hook("open", (path, "r", os.O_RDONLY))
    assert refusals == [code]


@pytest.fixture
def coordinate_environment(diagnostic, monkeypatch):
    """Exercise coordinate validation against an isolated invented environment."""
    values = {
        "DIAG_CHECKOUT_SHA": "1" * 40,
        "GITHUB_SHA": "1" * 40,
        "DIAG_WORKFLOW_SHA": "2" * 40,
        "DIAG_PR_HEAD_SHA": "3" * 40,
        "DIAG_PR_BASE_SHA": "4" * 40,
        "DIAG_MERGE_SHA": "5" * 40,
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_RUN_ID": "123456",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "PolicyEngine/microcosm",
    }
    # coordinates() consumes only environ. Keep the real process environment,
    # filesystem, dependency imports and bootstrap paths outside these cases.
    monkeypatch.setattr(diagnostic, "os", SimpleNamespace(environ=values))
    return values


@pytest.mark.parametrize("event", ("pull_request", "push"))
def test_coordinates_preserve_all_distinct_supplied_identities(
    diagnostic, coordinate_environment, event
):
    values = coordinate_environment
    values["GITHUB_EVENT_NAME"] = event
    original = dict(values)
    assert diagnostic.coordinates() == {
        "checkout_sha": "1" * 40,
        "event_sha": "1" * 40,
        "workflow_sha": "2" * 40,
        "pr_head_sha": "3" * 40,
        "pr_base_sha": "4" * 40,
        "event_merge_sha": "5" * 40,
        "github_run_id": "123456",
        "github_run_attempt": "2",
        "event": event,
        "repository": "PolicyEngine/microcosm",
    }
    assert values == original


@pytest.mark.parametrize("missing", (False, True))
def test_pull_request_coordinates_keep_absent_merge_metadata_empty(
    diagnostic, coordinate_environment, missing
):
    values = coordinate_environment
    if missing:
        del values["DIAG_MERGE_SHA"]
    else:
        values["DIAG_MERGE_SHA"] = ""
    result = diagnostic.coordinates()
    assert result["event"] == "pull_request"
    assert result["event_merge_sha"] == ""
    assert result["checkout_sha"] == result["event_sha"] == "1" * 40
    assert result["workflow_sha"] == "2" * 40
    assert result["pr_head_sha"] == "3" * 40
    assert result["pr_base_sha"] == "4" * 40
    assert values.get("DIAG_MERGE_SHA", "") == ""


@pytest.mark.parametrize("missing", (False, True))
def test_push_coordinates_allow_absent_pull_request_metadata(
    diagnostic, coordinate_environment, missing
):
    values = coordinate_environment
    values["GITHUB_EVENT_NAME"] = "push"
    for name in ("DIAG_PR_HEAD_SHA", "DIAG_PR_BASE_SHA", "DIAG_MERGE_SHA"):
        if missing:
            del values[name]
        else:
            values[name] = ""
    result = diagnostic.coordinates()
    assert result["event"] == "push"
    assert result["pr_head_sha"] == result["pr_base_sha"] == ""
    assert result["event_merge_sha"] == ""
    assert result["checkout_sha"] == result["event_sha"] == "1" * 40
    assert result["workflow_sha"] == "2" * 40


@pytest.mark.parametrize(
    ("event", "name"),
    (
        ("pull_request", "DIAG_CHECKOUT_SHA"),
        ("pull_request", "GITHUB_SHA"),
        ("pull_request", "DIAG_WORKFLOW_SHA"),
        ("pull_request", "DIAG_PR_HEAD_SHA"),
        ("pull_request", "DIAG_PR_BASE_SHA"),
        ("push", "DIAG_CHECKOUT_SHA"),
        ("push", "GITHUB_SHA"),
        ("push", "DIAG_WORKFLOW_SHA"),
    ),
)
def test_required_coordinate_hashes_still_refuse_when_absent(
    diagnostic, coordinate_environment, event, name
):
    values = coordinate_environment
    values["GITHUB_EVENT_NAME"] = event
    del values[name]
    with pytest.raises(diagnostic.RefusalError, match="^COORDINATES$"):
        diagnostic.coordinates()


@pytest.mark.parametrize(
    ("event", "name"),
    (
        ("pull_request", "DIAG_CHECKOUT_SHA"),
        ("pull_request", "GITHUB_SHA"),
        ("pull_request", "DIAG_WORKFLOW_SHA"),
        ("pull_request", "DIAG_PR_HEAD_SHA"),
        ("pull_request", "DIAG_PR_BASE_SHA"),
        ("pull_request", "DIAG_MERGE_SHA"),
        ("push", "DIAG_PR_HEAD_SHA"),
        ("push", "DIAG_PR_BASE_SHA"),
        ("push", "DIAG_MERGE_SHA"),
    ),
)
def test_nonempty_coordinate_hashes_still_require_valid_format(
    diagnostic, coordinate_environment, event, name
):
    values = coordinate_environment
    values["GITHUB_EVENT_NAME"] = event
    values[name] = "not-a-sha"
    with pytest.raises(diagnostic.RefusalError, match="^COORDINATES$"):
        diagnostic.coordinates()


@pytest.mark.parametrize(
    "value", ("null", "A" * 40, "f" * 39, "f" * 41, "f" * 40 + "\n")
)
def test_missing_merge_exception_does_not_accept_malformed_present_hashes(
    diagnostic, coordinate_environment, value
):
    coordinate_environment["DIAG_MERGE_SHA"] = value
    with pytest.raises(diagnostic.RefusalError, match="^COORDINATES$"):
        diagnostic.coordinates()


@pytest.mark.parametrize("event", ("pull_request", "push"))
def test_checkout_must_match_event_even_without_merge_metadata(
    diagnostic, coordinate_environment, event
):
    values = coordinate_environment
    values["GITHUB_EVENT_NAME"] = event
    values["DIAG_MERGE_SHA"] = ""
    values["DIAG_CHECKOUT_SHA"] = "6" * 40
    with pytest.raises(diagnostic.RefusalError, match="^CHECKOUT$"):
        diagnostic.coordinates()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("GITHUB_RUN_ID", None),
        ("GITHUB_RUN_ID", ""),
        ("GITHUB_RUN_ID", "not-a-run"),
        ("GITHUB_RUN_ATTEMPT", None),
        ("GITHUB_RUN_ATTEMPT", ""),
        ("GITHUB_RUN_ATTEMPT", "not-an-attempt"),
    ),
)
def test_coordinate_run_identity_remains_required(
    diagnostic, coordinate_environment, name, value
):
    if value is None:
        del coordinate_environment[name]
    else:
        coordinate_environment[name] = value
    with pytest.raises(diagnostic.RefusalError, match="^RUN_COORDINATES$"):
        diagnostic.coordinates()


@pytest.mark.parametrize(
    ("name", "value", "code"),
    (
        ("GITHUB_EVENT_NAME", None, "EVENT"),
        ("GITHUB_EVENT_NAME", "", "EVENT"),
        ("GITHUB_EVENT_NAME", "workflow_dispatch", "EVENT"),
        ("GITHUB_REPOSITORY", None, "REPOSITORY"),
        ("GITHUB_REPOSITORY", "invented/other", "REPOSITORY"),
    ),
)
def test_coordinate_event_and_repository_boundaries_remain_required(
    diagnostic, coordinate_environment, name, value, code
):
    if value is None:
        del coordinate_environment[name]
    else:
        coordinate_environment[name] = value
    with pytest.raises(diagnostic.RefusalError, match="^" + code + "$"):
        diagnostic.coordinates()
