"""The national release runs the gate battery in shadow (microcosm#1585 item 11).

``tools/build_us_fiscal_refresh_release.py`` arms the ``us_national`` scope
once the release id is known and finishes it in ``main``: after a finished
build (a verdict difference then fails it), at a dry run's deliberate stop
(which never fails it), or with today's exception. These tests drive those
seams with stand-in gates; no release is built.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build import GateResult
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.us_runtime.battery_bindings import (
    FRAME,
    us_flag_overrides,
    us_gate_specs,
)
from microcosm.build.us_runtime.gate_scopes import us_scoped_gate_manifest
from microcosm.build.us_runtime.gate_shadow import (
    GATE_BATTERY_SHADOW_DIRECTORY,
    US_GATE_SHADOW_SWITCH,
    GateBatteryShadowMismatchError,
    LegacyVerdict,
    current_gate_shadow,
    shadow_file_paths,
    shadow_value,
)
from test_support.microcosm_build.us_gate_shadow import (
    arm_us_tool_gate_shadows,
    load_us_gate_tool,
)


@pytest.fixture(autouse=True)
def _tool_shadows_on(monkeypatch) -> None:
    arm_us_tool_gate_shadows(monkeypatch)


@pytest.fixture
def release(monkeypatch):
    module = load_us_gate_tool("us_national")
    monkeypatch.setattr(module, "_ACTIVE_EMITTER", None, raising=False)
    monkeypatch.setattr(module, "_ACTIVE_PROGRESS", None, raising=False)
    yield module
    module._finish_national_gate_shadow(stopped_at="test teardown")


def _args(tmp_path, **overrides):
    values = {
        "out": tmp_path / "out",
        "dry_run_gates_report": None,
        "allow_input_mass_drift": False,
        "allow_ecps_parity_gaps": False,
        "allow_input_coverage_gaps": False,
        "allow_esi_premium_gaps": False,
        "allow_unpinned_feed": False,
        "allow_qrf_tail_concentration": False,
        "evidence_release": False,
        "input_mass_relative_tolerance": 0.5,
        "input_mass_minimum_reference_total": 1e9,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _specs():
    return {spec.check: spec for spec in us_gate_specs("us_national")}


def _target(release, reference):
    namespace, _, name = reference.partition(":")
    if namespace == "tool":
        return vars(release), name
    return vars(importlib.import_module(namespace)), name


_VERDICTS = {
    "result": LegacyVerdict.from_result,
    "receipt": lambda value: LegacyVerdict.from_failures(()),
    "failures": LegacyVerdict.from_failures,
    "failures_and_details": lambda value: LegacyVerdict.from_failures(value[0]),
    "result_and_details": lambda value: LegacyVerdict.from_result(value[0]),
    "failure_or_none": lambda value: LegacyVerdict.from_failures(
        [] if value is None else [value]
    ),
    "mapping": lambda value: LegacyVerdict.from_failures(
        () if value["passed"] else value["failures"]
    ),
}


def _stand_in_gates(monkeypatch, release, *, failing=()):
    """Every national gate the bindings call, standing in with a verdict."""

    for spec in _specs().values():
        target, name = _target(release, spec.gate)
        failed = spec.check in failing
        result = (
            GateResult(
                name=spec.legacy_name,
                passed=False,
                failures=(f"{spec.check}: stand-in.",),
            )
            if failed
            else GateResult(name=spec.legacy_name, passed=True)
        )
        lines = list(result.failures)
        value = {
            "result": result,
            "receipt": {"passed": True},
            "failures": lines,
            "failures_and_details": (lines, {}),
            "result_and_details": (result, {}),
            "failure_or_none": lines[0] if lines else None,
        }[spec.returns]
        monkeypatch.setitem(target, name, lambda *a, _value=value, **k: _value)


def _observe_every_entry(release, *, legacy_override=None):
    specs = _specs()
    for entry in us_scoped_gate_manifest("us_national").gates:
        spec = specs[entry.parameters["check"]]
        target, name = _target(release, spec.gate)
        evidence = EvidenceContext(
            frame=object() if spec.needs_frame else None,
            artifacts={key: object() for key in spec.artifact_keys},
        )
        today = (legacy_override or {}).get(entry.id) or (
            lambda target=target, name=name, spec=spec, evidence=evidence: target[name](
                *(
                    evidence.frame if source == FRAME else evidence.artifacts[source]
                    for source in spec.positional
                )
            )
        )
        shadow_value(
            entry.id,
            today,
            evidence,
            verdict=_VERDICTS[spec.returns],
            refusals=spec.refusals,
        )


def _comparison(directory):
    _, comparison = shadow_file_paths(directory, "us_national")
    return json.loads(comparison.read_text())


def test_main_finishes_the_shadow_a_build_armed(tmp_path, monkeypatch, release):
    args = _args(tmp_path)
    _stand_in_gates(monkeypatch, release)

    def build(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        _observe_every_entry(release)
        return None

    monkeypatch.setattr(release, "_main", build)
    release.main([])
    payload = _comparison(release._national_gate_shadow_directory(args))
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is False
    assert payload["phases_recorded"] == ["preflight", "assembled", "terminal"]
    assert payload["release_id"] == "microcosm-us-test"


def test_each_release_keeps_its_own_shadow_files(tmp_path, release) -> None:
    national = release._national_gate_shadow_directory(_args(tmp_path))
    evidence = release._national_gate_shadow_directory(
        _args(tmp_path, evidence_release=True)
    )
    shared = (tmp_path / "out").resolve() / GATE_BATTERY_SHADOW_DIRECTORY
    assert national == shared / release.US_NATIONAL_RELEASE_ID
    assert evidence == shared / release.US_EVIDENCE_RELEASE_ID


def test_a_verdict_difference_fails_a_finished_build(tmp_path, monkeypatch, release):
    args = _args(tmp_path)
    _stand_in_gates(monkeypatch, release, failing={"pregnancy_signal"})
    passing = GateResult(name="pregnancy_signal", passed=True)

    def build(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        _observe_every_entry(
            release, legacy_override={"us_national_pregnancy": lambda: passing}
        )
        return None

    monkeypatch.setattr(release, "_main", build)
    with pytest.raises(GateBatteryShadowMismatchError, match="us_national_pregnancy"):
        release.main([])


class _Recorder:
    """A stand-in staging run or emitter that records each outcome it is told."""

    available = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def complete(self) -> None:
        self.calls.append(("complete", None))

    def fail(self, error: BaseException, **_: object) -> None:
        self.calls.append(("fail", type(error).__name__))


@pytest.mark.parametrize("difference", [False, True])
def test_a_finished_builds_staging_run_keeps_the_outcome_it_recorded(
    tmp_path, monkeypatch, release, difference
):
    # Review of #1635: _main completes a finished build's staging run before
    # main finishes the shadow. A difference raised there must not rewrite
    # that run as failed. The program's emitter records one outcome.
    args = _args(tmp_path)
    _stand_in_gates(
        monkeypatch, release, failing={"pregnancy_signal"} if difference else ()
    )
    passing = GateResult(name="pregnancy_signal", passed=True)
    progress, emitter = _Recorder(), _Recorder()
    monkeypatch.setattr(release, "_ACTIVE_EMITTER", emitter)

    def build(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        _observe_every_entry(
            release,
            legacy_override=(
                {"us_national_pregnancy": lambda: passing} if difference else None
            ),
        )
        release._ACTIVE_PROGRESS = progress
        progress.complete()
        return None

    monkeypatch.setattr(release, "_main", build)
    if difference:
        with pytest.raises(
            GateBatteryShadowMismatchError, match="us_national_pregnancy"
        ):
            release.main([])
    else:
        release.main([])
    assert progress.calls == [("complete", None)]
    assert emitter.calls == [
        ("fail", "GateBatteryShadowMismatchError") if difference else ("complete", None)
    ]


def test_a_build_error_still_fails_the_staging_run_and_the_emitter(
    tmp_path, monkeypatch, release
):
    # Today's handler is unchanged: an error inside _main fails both, once.
    args = _args(tmp_path)
    progress, emitter = _Recorder(), _Recorder()
    monkeypatch.setattr(release, "_ACTIVE_EMITTER", emitter)

    def refused(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        release._ACTIVE_PROGRESS = progress
        raise RuntimeError("Release gates failed: something upstream")

    monkeypatch.setattr(release, "_main", refused)
    with pytest.raises(RuntimeError, match="something upstream"):
        release.main([])
    assert progress.calls == [("fail", "RuntimeError")]
    assert emitter.calls == [("fail", "RuntimeError")]


def test_a_dry_run_stops_on_purpose_and_never_fails(tmp_path, monkeypatch, release):
    report = tmp_path / "dry" / "gates.json"
    args = _args(tmp_path, dry_run_gates_report=report)
    _stand_in_gates(monkeypatch, release)

    def dry_run(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        return 0

    monkeypatch.setattr(release, "_main", dry_run)
    with pytest.raises(SystemExit) as stopped:
        release.main([])
    assert stopped.value.code == 0
    # Beside the dry run's own report; a dry run creates no --out.
    payload = _comparison(tmp_path / "dry" / "gates.gate_battery_shadow")
    assert payload["run_completed"] is False
    assert payload["stopped_at"].startswith("dry run")
    assert payload["verdict_difference"] is False
    assert not args.out.exists()


def test_todays_exception_propagates_and_the_shadow_records_it(
    tmp_path, monkeypatch, release
):
    args = _args(tmp_path)

    def refused(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        # Past the certified-directory refusal, where _main lets it write.
        release._release_national_gate_shadow_files()
        raise RuntimeError("Release gates failed: something upstream")

    monkeypatch.setattr(release, "_main", refused)
    with pytest.raises(RuntimeError, match="something upstream"):
        release.main([])
    payload = _comparison(release._national_gate_shadow_directory(args))
    assert payload["stopped_by"]["type"] == "RuntimeError"
    assert payload["run_completed"] is False


def test_a_run_refused_before_its_outputs_leaves_earlier_files_alone(
    tmp_path, monkeypatch, release
):
    args = _args(tmp_path)
    report, comparison = shadow_file_paths(
        release._national_gate_shadow_directory(args), "us_national"
    )
    comparison.parent.mkdir(parents=True)
    comparison.write_text('{"the certified build": true}')
    report.write_text("{}")

    def refused(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        raise RuntimeError("the release directory already carries a certified build")

    monkeypatch.setattr(release, "_main", refused)
    with pytest.raises(RuntimeError, match="already carries"):
        release.main([])
    assert json.loads(comparison.read_text()) == {"the certified build": True}
    assert report.read_text() == "{}"


def test_main_lets_the_shadow_write_only_after_the_certified_dir_refusal(
    release,
) -> None:
    tree = ast.parse(Path(release.__file__).read_text())
    main_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_main"
    )
    statements = [ast.unparse(statement) for statement in main_fn.body]
    refusal = statements.index("_refuse_certified_release_dir_reuse(release_dir)")
    assert statements[refusal + 1] == "_release_national_gate_shadow_files()"
    starts = [
        index
        for index, statement in enumerate(statements)
        if statement.startswith("_start_national_gate_shadow(")
    ]
    assert len(starts) == 1 and starts[0] < refusal


def test_with_the_switch_off_main_arms_no_shadow(tmp_path, monkeypatch, release):
    monkeypatch.setenv(US_GATE_SHADOW_SWITCH, "off")
    args = _args(tmp_path)
    armed = []

    def build(argv):
        release._start_national_gate_shadow(args, release_id="microcosm-us-test")
        armed.append(current_gate_shadow("us_national_pregnancy"))
        return None

    monkeypatch.setattr(release, "_main", build)
    release.main([])
    assert armed == [None]
    assert not (args.out / GATE_BATTERY_SHADOW_DIRECTORY).exists()


def test_waivers_and_flag_overrides_are_recorded_by_entry(tmp_path, release):
    gates = us_scoped_gate_manifest("us_national")
    args = _args(
        tmp_path,
        allow_input_mass_drift=True,
        allow_esi_premium_gaps=True,
        input_mass_relative_tolerance=0.2,
    )
    waivers = release._national_gate_waivers(args, gates)
    assert waivers["us_national_input_mass_reference"] == ["--allow-input-mass-drift"]
    assert waivers["us_national_esi_premiums_anchor"] == ["--allow-esi-premium-gaps"]
    assert "us_national_ecps_parity" not in waivers
    overrides = us_flag_overrides(gates, us_gate_specs("us_national"), args)
    assert overrides == {
        entry_id: {"relative_tolerance": {"declared": 0.5, "runtime": 0.2}}
        for entry_id in (
            "us_national_input_mass_reference",
            "us_national_export_input_mass",
        )
    }
    evidence = release._national_gate_waivers(
        _args(tmp_path, evidence_release=True), gates
    )
    assert evidence["us_national_release_gates"] == ["--evidence-release"]
    assert "us_national_pregnancy" not in evidence
