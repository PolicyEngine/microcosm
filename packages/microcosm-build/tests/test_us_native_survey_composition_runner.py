"""Pure parts of ``tools/run_us_native_survey_composition.py``.

These tests never read survey or PUF microdata, fit a model or run a graph.
They pin the runner's reconstruction of the reviewed development profile, its
refusals and its aggregate-only bookkeeping. The positive composition itself is
exercised by the guarded actual run, whose receipt lives outside the repository.
"""

from __future__ import annotations

import copy
import importlib.util
import types
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools/run_us_native_survey_composition.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("composition_runner", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def host():
    from microcosm.build.us_runtime import graph_atomic_survey_financial

    return graph_atomic_survey_financial


def test_sanitized_error_shows_bare_codes(runner):
    error = ValueError("PUF_PACKAGED_ROUTES")
    assert runner.sanitized_error(error) == {
        "type": "ValueError",
        "message": "PUF_PACKAGED_ROUTES",
        "causes": [],
    }


def test_sanitized_error_withholds_messages_that_may_quote_values(runner):
    try:
        try:
            raise KeyError("household 12345 value 6789.5")
        except KeyError as inner:
            raise ValueError("SPM_COLUMNS_INCOMPLETE: row 17 had 3.25") from inner
    except ValueError as error:
        result = runner.sanitized_error(error)
    assert "12345" not in str(result) and "3.25" not in str(result)
    assert result["message"].startswith("message withheld")
    assert "SPM_COLUMNS_INCOMPLETE" in result["message"]
    assert result["causes"] == [{"type": "KeyError", "message": None}]


def test_completion_options_reproduce_the_reviewed_declarations(runner, host):
    properties, children = runner.completion_options(
        runner.DEVELOPMENT_PROFILE["financial"], host
    )
    assert properties.completion_routing is True
    assert properties.n_estimators == 2
    assert children.scope == "candidate"
    assert children.document()["source_observation_claim"] is False


@pytest.mark.parametrize(
    ("section", "key", "value", "code"),
    [
        ("property_income", "rtol", 1e-9, "PROPERTY_DECLARATION_CHANGED"),
        ("property_income", "n_estimators", 3, "PROPERTY_DECLARATION_CHANGED"),
        ("child_property", "scenario_id", "other", "CHILD_DECLARATION_CHANGED"),
        ("child_property", "scope", "test", "CHILD_SCOPE_MUST_REMAIN_CANDIDATE"),
    ],
)
def test_changed_declared_options_refuse(runner, host, section, key, value, code):
    profile = copy.deepcopy(runner.DEVELOPMENT_PROFILE["financial"])
    profile[section][key] = value
    with pytest.raises(runner.CompositionRefusedError, match=code):
        runner.completion_options(profile, host)


def test_argument_builders_refuse_unknown_resume(runner):
    profile = runner.DEVELOPMENT_PROFILE
    with pytest.raises(runner.CompositionRefusedError, match="RESUME"):
        runner.puf_arguments(profile, donors={}, resume="force")
    with pytest.raises(runner.CompositionRefusedError, match="RESUME"):
        runner.enrichment_arguments(profile, {}, resume="skip")


def test_puf_and_enrichment_arguments_keep_the_development_profile(runner):
    profile = runner.DEVELOPMENT_PROFILE
    puf = runner.puf_arguments(profile, donors={"a": Path("a")}, resume="auto")
    assert puf == {
        "donor_sources": {"a": Path("a")},
        "seed": 578,
        "n_estimators": 2,
        "zero_atol": 1e-8,
        "fixture_definition": None,
        "resume": "auto",
    }
    policy = {"acs_profile": object(), "asec_scope_policy": object()}
    enrichment = runner.enrichment_arguments(profile, policy, resume="require")
    assert enrichment["spm_acs_profile"] is policy["acs_profile"]
    assert enrichment["spm_asec_scope_policy"] is policy["asec_scope_policy"]
    assert enrichment["original_application_seed"] == 579 != puf["seed"]
    assert enrichment["canonical_state_input"] is True
    assert enrichment["spm_outside_role_placeholder"] is False
    # Optional successors the development profile does not admit stay off.
    assert enrichment["immigration_transfer"] is None
    for name in (
        "health_completion",
        "demographic_inputs",
        "race_hispanic_inputs",
        "full_original_amount_donors",
    ):
        assert enrichment[name] is False


def test_spm_policy_builds_the_closed_asec_scope_policy(runner):
    from microcosm.build.us_runtime import current_survey_spm_source as source

    policy = runner.spm_policy(runner.DEVELOPMENT_PROFILE, source)
    assert policy["asec_scope_policy"] is source.ASEC_2025_INCOME_2024_SPM_POLICY
    assert policy["acs_profile"].admit_modeled is True
    changed = copy.deepcopy(runner.DEVELOPMENT_PROFILE)
    changed["enrichment"]["spm_asec_scope_policy"] = "OTHER"
    with pytest.raises(runner.CompositionRefusedError, match="SPM_SCOPE_POLICY"):
        runner.spm_policy(changed, source)


def test_donor_sources_bind_packaged_pins_by_bytes(runner, tmp_path):
    from microcosm.build.us_runtime import graph_survey_puf55 as puf

    missing = tmp_path / "missing.csv"
    with pytest.raises(runner.CompositionRefusedError, match="PUF_FILE_MISSING"):
        runner.donor_sources(puf, main=missing, demographic=missing)
    wrong = tmp_path / "wrong.csv"
    wrong.write_bytes(b"invented,columns\n1,2\n")
    with pytest.raises(runner.CompositionRefusedError, match="PUF_FILE_SIZE"):
        runner.donor_sources(puf, main=wrong, demographic=wrong)


def _args(runner, tmp_path, **changes):
    values = dict(
        output_root=str(tmp_path / "out"),
        resume="auto",
        reuse_store=False,
        min_available_gib=0.0,
        min_free_disk_gib=0.0,
        allow_other_native_runs=False,
    )
    values.update(changes)
    return types.SimpleNamespace(**values)


def test_admission_refusals(runner, tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    monkeypatch.setattr(runner, "other_native_runs", lambda: [])
    assert runner.admission(_args(runner, tmp_path))["refusals"] == []
    refusals = runner.admission(
        _args(runner, tmp_path, min_available_gib=1e9, min_free_disk_gib=1e9)
    )["refusals"]
    assert refusals == ["ADMISSION_AVAILABLE_MEMORY", "ADMISSION_FREE_DISK"]
    (tmp_path / "out/graph-store").mkdir(parents=True)
    cold = runner.admission(_args(runner, tmp_path))["refusals"]
    assert cold == ["ADMISSION_COLD_STORE_EXISTS"]
    assert runner.admission(_args(runner, tmp_path, reuse_store=True))["refusals"] == []
    missing = runner.admission(_args(runner, tmp_path / "elsewhere", resume="require"))[
        "refusals"
    ]
    assert missing == ["ADMISSION_REQUIRED_STORE_MISSING"]
    monkeypatch.setattr(
        runner, "other_native_runs", lambda: [{"pid": 1, "markers": ["harness19"]}]
    )
    assert runner.admission(_args(runner, tmp_path / "x"))["refusals"] == [
        "ADMISSION_OTHER_NATIVE_RUN"
    ]


def test_other_native_runs_never_counts_this_process(runner):
    pytest.importorskip("psutil")
    import os

    assert all(row["pid"] != os.getpid() for row in runner.other_native_runs())


class _Node:
    def __init__(self, hit, key="k", impl="i", artifacts=(), opaque=()):
        self.hit = hit
        self.key = key
        self.kernel_impl_hash = impl
        self.artifacts = artifacts
        self.opaque_artifacts = opaque


class _Receipt:
    def __init__(self, seconds):
        self.wall_time_s = seconds


def _run(nodes, *, key="manifest"):
    manifest = types.SimpleNamespace(
        node=lambda name: nodes[name],
        receipts={name: _Receipt(float(i + 1)) for i, name in enumerate(nodes)},
        key=key,
    )
    return types.SimpleNamespace(
        compiled=types.SimpleNamespace(order=tuple(nodes)), manifest=manifest
    )


def test_inherited_records_must_hit_with_unchanged_keys(runner):
    parent = _run({"a": _Node(False), "b": _Node(False)})
    child = _run({"a": _Node(True), "b": _Node(True), "c": _Node(False)})
    runner.assert_inherited(parent, child)
    runner.check_new_nodes(child, inherited=("a", "b"), required=False)
    with pytest.raises(runner.CompositionRefusedError, match="REQUIRED_NODE_MISSED"):
        runner.check_new_nodes(child, inherited=("a", "b"), required=True)
    changed = _run({"a": _Node(True, key="other"), "b": _Node(True)})
    with pytest.raises(runner.CompositionRefusedError, match="INHERITED_RECORD"):
        runner.assert_inherited(parent, changed)
    missed = _run({"a": _Node(False), "b": _Node(True)})
    with pytest.raises(runner.CompositionRefusedError, match="INHERITED_RECORD"):
        runner.assert_inherited(parent, missed)
    with pytest.raises(runner.CompositionRefusedError, match="INHERITED_NODE_MISSED"):
        runner.check_new_nodes(missed, inherited=("a",), required=False)


def test_graph_counts_are_aggregates(runner):
    run = _run({"a": _Node(True), "b": _Node(False), "c": _Node(True)})
    counts = runner.graph_counts(run, inherited=("a",))
    assert counts == {
        "nodes": 3,
        "inherited_nodes": 1,
        "new_nodes": 2,
        "store_hits": 2,
        "new_node_hits": 1,
        "node_loop_wall_seconds": 6.0,
        "manifest_key": "manifest",
    }
    assert runner.node_wall_times(run, exclude=("c",)) == [["b", 2.0], ["a", 1.0]]


def test_check_ledger_records_failures_and_continues(runner):
    records = {}
    ledger = runner.CheckLedger(lambda name, value: records.__setitem__(name, value))
    with ledger.check("first"):
        runner.require(False, "FIRST_FAILED")
    with ledger.check("second"):
        pass
    assert ledger.passed == ["second"]
    assert [row["check"] for row in ledger.failed] == ["first"]
    assert ledger.failed[0]["message"] == "NATIVE_COMPOSITION_FIRST_FAILED"
    assert records["checks"]["failed"] == ledger.failed
    with pytest.raises(KeyboardInterrupt):
        with ledger.check("interrupt"):
            raise KeyboardInterrupt
