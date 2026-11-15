"""The US gate battery in shadow beside today's gates (microcosm#1585 item 11).

PR 2 of 4. Today's gates stay authoritative; each tool evaluates the declared
entry beside each gate, compares verdicts, and writes the scope's report and
the comparison as shadow files. These tests hold the comparator's rule (the
waiver proposal on microcosm#1609), the shadow's never-raise-mid-run
discipline, and the assembly of phases from stage outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    EvidenceContext,
    GateOutcome,
    GateStatus,
)
from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.battery_bindings import (
    FRAME,
    USCheckBinding,
    USGateSpec,
    us_check_binding,
)
from microcosm.build.us_runtime.gate_shadow import (
    US_GATE_SHADOW_SWITCH,
    GateBatteryShadow,
    GateBatteryShadowMismatchError,
    LegacyStatus,
    LegacyVerdict,
    compare_verdicts,
    legacy_parameter_overrides,
    shadow_file_paths,
    tool_gate_shadows_enabled,
)


def _manifest(gates: list[dict], phases: list[str] | None = None) -> GatesManifest:
    return GatesManifest.from_mapping(
        {
            "version": 1,
            "country": "us",
            "policy": "test shadow scope",
            "phases": phases or ["assembled", "terminal"],
            "gates": gates,
        }
    )


def _entry(id_: str, check: str, phase: str = "assembled", **overrides) -> dict:
    entry = {
        "id": id_,
        "gate": "stage_health",
        "phase": phase,
        "criticality": "release_blocking",
        "parameters": {"check": check},
    }
    entry.update(overrides)
    return entry


def _gate(name: str):
    """A legacy gate over an artifact: fails when the value exceeds ``cap``."""

    def gate(value, *, cap=1.0):
        if value > cap:
            return GateResult(
                name=name,
                passed=False,
                failures=(f"{name}: {value} exceeds {cap}.",),
                details={"value": value},
            )
        return GateResult(name=name, passed=True, details={"value": value})

    return gate


ALPHA = _gate("us_alpha_signal")
BETA = _gate("us_beta_signal")


#: The tool namespace the specs below resolve "tool:" references in, as a
#: build tool passes its ``globals()``.
TOOL = {"ALPHA": ALPHA, "BETA": BETA, "CRASH": None}


def _crashing_tool_gate(value, *, cap=1.0):
    raise KeyError("missing column 'x'")


TOOL["CRASH"] = _crashing_tool_gate


def _registry():
    specs = (
        USGateSpec(
            check="alpha",
            legacy_name="us_alpha_signal",
            gate="tool:ALPHA",
            positional=("alpha_value",),
            parameters=frozenset({"cap"}),
        ),
        USGateSpec(
            check="beta",
            legacy_name="us_beta_signal",
            gate="tool:BETA",
            arguments={"value": "beta_value"},
            parameters=frozenset({"cap"}),
        ),
        USGateSpec(
            check="crash",
            legacy_name="us_crash_signal",
            gate="tool:CRASH",
            positional=("alpha_value",),
            parameters=frozenset({"cap"}),
        ),
    )
    return us_check_binding("stage_health", specs, tool=TOOL)


def _shadow(tmp_path, gates, **kwargs) -> GateBatteryShadow:
    report, comparison = shadow_file_paths(tmp_path, "us_test")
    return GateBatteryShadow(
        gates,
        scope="us_test",
        release_id="us-test-build",
        report_path=report,
        comparison_path=comparison,
        registry=kwargs.pop("registry", _registry()),
        **kwargs,
    )


def _two_phase_gates() -> GatesManifest:
    return _manifest(
        [
            _entry("us_test_alpha", "alpha", parameters={"check": "alpha", "cap": 1.0}),
            _entry("us_test_beta", "beta", parameters={"check": "beta", "cap": 1.0}),
            _entry(
                "us_test_final",
                "alpha",
                phase="terminal",
                parameters={"check": "alpha", "cap": 1.0},
            ),
        ]
    )


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------


def _outcome(status: GateStatus, *, failures=(), details=None, reason=None):
    entry = _two_phase_gates().gates[0]
    result = None
    if status in (GateStatus.PASSED, GateStatus.FAILED):
        result = GateResult(
            name="stage_health",
            passed=status is GateStatus.PASSED,
            failures=tuple(failures),
            details=details or {},
        )
    return GateOutcome(entry=entry, status=status, result=result, reason=reason)


CRASH = {"evaluation_error": {"type": "KeyError", "message": "'x'"}}


@pytest.mark.parametrize(
    ("legacy", "outcome", "match"),
    [
        (LegacyVerdict.from_failures(()), _outcome(GateStatus.PASSED), True),
        (
            LegacyVerdict.from_failures(("a", "b")),
            _outcome(GateStatus.FAILED, failures=("a", "b")),
            True,
        ),
        (
            LegacyVerdict.from_failures(("a", "b")),
            _outcome(GateStatus.FAILED, failures=("b", "a")),
            False,
        ),
        (
            LegacyVerdict.from_failures(()),
            _outcome(GateStatus.FAILED, failures=("a",)),
            False,
        ),
        (
            LegacyVerdict.from_failures(("a",)),
            _outcome(GateStatus.PASSED),
            False,
        ),
        # A crash in today's code and the battery's fail-closed crash are the
        # same verdict.
        (
            LegacyVerdict.from_exception(KeyError("x")),
            _outcome(GateStatus.FAILED, failures=("crashed",), details=CRASH),
            True,
        ),
        (
            LegacyVerdict.from_exception(KeyError("x")),
            _outcome(GateStatus.FAILED, failures=("measured",)),
            False,
        ),
        (
            LegacyVerdict.from_exception(KeyError("x")),
            _outcome(GateStatus.PASSED),
            False,
        ),
        (
            LegacyVerdict.from_failures(("crashed",)),
            _outcome(GateStatus.FAILED, failures=("crashed",), details=CRASH),
            False,
        ),
        (
            LegacyVerdict.not_run("the stage did not run"),
            _outcome(GateStatus.EVIDENCE_ABSENT, reason="missing evidence: frame"),
            True,
        ),
        (
            LegacyVerdict.not_run("the stage did not run"),
            _outcome(GateStatus.PASSED),
            False,
        ),
        (
            LegacyVerdict.from_failures(()),
            _outcome(GateStatus.EVIDENCE_ABSENT, reason="missing evidence: frame"),
            False,
        ),
    ],
)
def test_the_comparator_compares_verdicts(legacy, outcome, match) -> None:
    difference = compare_verdicts(legacy, outcome)
    assert (difference is None) is match
    if not match:
        assert isinstance(difference, str) and difference


def test_a_legacy_verdict_is_consistent() -> None:
    with pytest.raises(ValueError):
        LegacyVerdict(status=LegacyStatus.FAILED)
    with pytest.raises(ValueError):
        LegacyVerdict(status=LegacyStatus.PASSED, failures=("a",))
    with pytest.raises(ValueError):
        LegacyVerdict(status=LegacyStatus.ERROR)
    with pytest.raises(ValueError):
        LegacyVerdict(status=LegacyStatus.NOT_RUN)
    with pytest.raises(TypeError):
        LegacyVerdict.from_failures("one line")
    with pytest.raises(TypeError):
        LegacyVerdict.from_result(["a"])
    assert LegacyVerdict.from_result(
        GateResult(name="x", passed=False, failures=("a",))
    ) == LegacyVerdict.from_failures(["a"])


# ---------------------------------------------------------------------------
# Observing gates
# ---------------------------------------------------------------------------


def test_a_matching_run_writes_both_files_and_passes(tmp_path) -> None:
    gates = _two_phase_gates()
    with _shadow(tmp_path, gates) as shadow:
        result = shadow.gate(
            "us_test_alpha",
            lambda: ALPHA(0.5),
            EvidenceContext(artifacts={"alpha_value": 0.5}),
            site="after the alpha stage",
        )
        assert result.name == "us_alpha_signal"
        shadow.gate(
            "us_test_beta",
            lambda: BETA(2.0),
            lambda: EvidenceContext(artifacts={"beta_value": 2.0}),
        )
        shadow.gate(
            "us_test_final",
            lambda: ALPHA(0.1),
            EvidenceContext(artifacts={"alpha_value": 0.1}),
        )
    comparison = json.loads(shadow.comparison_path.read_text())
    assert comparison["run_completed"] is True
    assert comparison["verdict_difference"] is False
    assert comparison["phases_recorded"] == ["assembled", "terminal"]
    assert [row["id"] for row in comparison["entries"]] == [
        "us_test_alpha",
        "us_test_beta",
        "us_test_final",
    ]
    beta = comparison["entries"][1]
    assert beta["legacy"]["status"] == "failed"
    assert beta["battery"]["status"] == "failed"
    assert beta["battery"]["failures"] == beta["legacy"]["failures"]
    report_bytes = shadow.report_path.read_bytes()
    assert comparison["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    report = json.loads(report_bytes)
    assert report["release_candidate"] is False
    assert report["shippable"] is False
    assert report["gates"]["us_test_beta"]["status"] == "failed"
    assert report["blocked_at_phase"] is None
    assert comparison["gates_manifest_sha256"] == report["gates_manifest_sha256"]


def test_a_waived_gate_that_fails_in_both_is_a_match(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates, waivers={"us_test_beta": ["--allow-beta"]})
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    # Today's code records the failure and goes on: the flag waives it.
    shadow.gate(
        "us_test_beta",
        lambda: BETA(3.0),
        EvidenceContext(artifacts={"beta_value": 3.0}),
    )
    shadow.gate(
        "us_test_final",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    payload = shadow.finish()
    assert payload["verdict_difference"] is False
    assert payload["waivers"] == {"us_test_beta": ["--allow-beta"]}
    (beta,) = [row for row in payload["entries"] if row["id"] == "us_test_beta"]
    assert beta["waived_by"] == ["--allow-beta"] and beta["match"] is True


def test_a_crash_in_both_is_a_match_and_todays_exception_propagates(tmp_path) -> None:
    gates = _manifest(
        [_entry("us_test_crash", "crash"), _entry("us_test_alpha", "alpha")]
    )
    shadow = _shadow(tmp_path, gates)
    error = KeyError("missing column 'x'")

    def todays_gate():
        raise error

    with pytest.raises(KeyError) as raised:
        with shadow:
            shadow.gate(
                "us_test_crash",
                todays_gate,
                EvidenceContext(artifacts={"alpha_value": 0.5}),
            )
    assert raised.value is error
    payload = json.loads(shadow.comparison_path.read_text())
    (crash,) = payload["entries"]
    assert crash["legacy"]["status"] == "error"
    assert crash["battery"]["evaluation_error"]["type"] == "KeyError"
    assert crash["match"] is True
    assert payload["run_completed"] is False
    assert payload["stopped_by"]["type"] == "KeyError"
    # The crash blocks: its phase is recorded with the rest unreached.
    report = json.loads(shadow.report_path.read_text())
    assert report["blocked_at_phase"] == "assembled"
    assert report["gates"]["us_test_alpha"]["status"] == "unreached"


def test_a_difference_fails_the_finished_run_after_both_files(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates)
    # Today's code passes at a looser cap than the spec declares.
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(1.5, cap=2.0),
        EvidenceContext(artifacts={"alpha_value": 1.5}),
    )
    shadow.gate(
        "us_test_beta",
        lambda: BETA(0.5),
        EvidenceContext(artifacts={"beta_value": 0.5}),
    )
    shadow.gate(
        "us_test_final",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    with pytest.raises(GateBatteryShadowMismatchError, match="us_test_alpha") as raised:
        shadow.finish()
    assert raised.value.comparison_path.is_file()
    payload = json.loads(raised.value.comparison_path.read_text())
    assert payload["verdict_difference"] is True
    assert [row["id"] for row in payload["differences"]] == ["us_test_alpha"]
    assert shadow.report_path.is_file()


def test_a_runtime_override_explains_its_difference(tmp_path) -> None:
    gates = _two_phase_gates()
    overrides = legacy_parameter_overrides(
        {"us_test_alpha": {"cap": 1.0}}, {"us_test_alpha": {"cap": 2.0}}
    )
    assert overrides == {"us_test_alpha": {"cap": {"declared": 1.0, "runtime": 2.0}}}
    shadow = _shadow(tmp_path, gates, overrides=overrides)
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(1.5, cap=2.0),
        EvidenceContext(artifacts={"alpha_value": 1.5}),
    )
    shadow.gate(
        "us_test_beta",
        lambda: BETA(0.5),
        EvidenceContext(artifacts={"beta_value": 0.5}),
    )
    shadow.gate(
        "us_test_final",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    payload = shadow.finish()
    assert payload["verdict_difference"] is False
    assert [row["id"] for row in payload["explained_differences"]] == ["us_test_alpha"]


def test_an_unwired_entry_fails_a_finished_run(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates)
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    shadow.gate(
        "us_test_beta",
        lambda: BETA(0.5),
        EvidenceContext(artifacts={"beta_value": 0.5}),
    )
    with pytest.raises(GateBatteryShadowMismatchError, match="us_test_final"):
        shadow.finish()
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["not_observed"] == ["us_test_final"]
    assert payload["phases_recorded"] == ["assembled"]


def test_a_deliberate_stop_records_what_ran_and_never_raises(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates)
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    shadow.gate(
        "us_test_beta",
        lambda: BETA(0.5),
        EvidenceContext(artifacts={"beta_value": 0.5}),
    )
    payload = shadow.finish(stopped_at="dry run: stopped before target materialization")
    assert payload["run_completed"] is False
    assert payload["stopped_at"].startswith("dry run")
    assert payload["not_observed"] == ["us_test_final"]
    assert payload["verdict_difference"] is False
    # Leaving the context manager after a deliberate finish changes nothing.
    shadow.__exit__(None, None, None)


def test_a_stop_inside_a_phase_names_the_last_blocking_entry(tmp_path) -> None:
    gates = _manifest(
        [
            _entry("us_test_first", "alpha"),
            _entry("us_test_second", "beta"),
            _entry("us_test_third", "alpha"),
            _entry("us_test_final", "alpha", phase="terminal"),
        ]
    )
    shadow = _shadow(tmp_path, gates)
    error = RuntimeError("Release gates failed: beta")
    with pytest.raises(RuntimeError):
        with shadow:
            shadow.gate(
                "us_test_first",
                lambda: ALPHA(0.5),
                EvidenceContext(artifacts={"alpha_value": 0.5}),
            )
            result = shadow.gate(
                "us_test_second",
                lambda: BETA(4.0),
                EvidenceContext(artifacts={"beta_value": 4.0}),
            )
            if not result.passed:
                raise error
    report = json.loads(shadow.report_path.read_text())
    assert report["blocked_at_phase"] == "assembled"
    assert report["gates"]["us_test_second"]["status"] == "failed"
    assert report["gates"]["us_test_third"]["status"] == "unreached"
    assert "us_test_second" in report["gates"]["us_test_third"]["reason"]
    assert report["gates"]["us_test_final"]["status"] == "unreached"
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["phases_recorded"] == ["assembled"]
    assert payload["stopped_by"] == {
        "type": "RuntimeError",
        "message": "Release gates failed: beta",
    }
    # The run took the stopped phase: it marks unreached no entry it
    # evaluated (#1609), and no observations are missing.
    assert payload["shadow_errors"] == []
    assert payload["observations_missing"] == []


def test_a_stop_that_no_gate_explains_leaves_its_phase_unrecorded(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates)
    with pytest.raises(OSError):
        with shadow:
            shadow.gate(
                "us_test_alpha",
                lambda: ALPHA(0.5),
                EvidenceContext(artifacts={"alpha_value": 0.5}),
            )
            raise OSError("disk full")
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["phases_recorded"] == []
    assert payload["report_file"] is None
    assert not shadow.report_path.exists()


def _stopping_gates() -> GatesManifest:
    return _manifest(
        [
            _entry("us_test_first", "alpha"),
            _entry("us_test_second", "beta"),
            _entry("us_test_third", "alpha"),
            _entry("us_test_final", "alpha", phase="terminal"),
        ]
    )


def _broken_context():
    raise ValueError("cannot build evidence")


def _stop_on_a_failed_beta(shadow: GateBatteryShadow) -> None:
    """Observe a failing second entry and stop as today's code would."""

    with pytest.raises(RuntimeError, match="Release gates failed: beta"):
        with shadow:
            result = shadow.gate(
                "us_test_second",
                lambda: BETA(4.0),
                EvidenceContext(artifacts={"beta_value": 4.0}),
            )
            if not result.passed:
                raise RuntimeError("Release gates failed: beta")


def test_a_stop_never_names_unreached_a_gate_today_reached(tmp_path) -> None:
    # Review of #1609: a stopped phase names unreached the entries without an
    # outcome. Today's code reached the first entry; the shadow only failed to
    # build its evidence, so the phase stays unrecorded.
    shadow = _shadow(tmp_path, _stopping_gates())
    shadow.gate("us_test_first", lambda: ALPHA(0.5), _broken_context)
    _stop_on_a_failed_beta(shadow)
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["phases_recorded"] == []
    assert payload["report_file"] is None
    assert not shadow.report_path.exists()
    assert payload["not_observed"] == [
        "us_test_first",
        "us_test_third",
        "us_test_final",
    ]
    (error,) = payload["shadow_errors"]
    assert "[us_test_first]" in error and "cannot build evidence" in error


def test_a_later_process_never_names_unreached_what_an_earlier_one_reached(
    tmp_path,
) -> None:
    gates = _stopping_gates()
    earlier = _shadow(tmp_path / "earlier", gates)
    earlier.gate("us_test_first", lambda: ALPHA(0.5), _broken_context)
    saved = tmp_path / "earlier.observations.json"
    earlier.save_observations(saved)
    stored = json.loads(saved.read_text())
    assert stored["reached"] == ["us_test_first"]
    assert stored["observations"] == []

    later = _shadow(tmp_path / "later", gates)
    later.load_observations(saved)
    _stop_on_a_failed_beta(later)
    payload = json.loads(later.comparison_path.read_text())
    assert payload["phases_recorded"] == []
    assert not later.report_path.exists()
    (error,) = payload["shadow_errors"]
    assert "cannot build evidence" in error


@pytest.mark.parametrize("missing", ["a stage saved nothing", "a file is torn"])
def test_missing_observations_leave_the_stopped_phase_unrecorded(
    tmp_path, missing
) -> None:
    # The first entry may have been reached by the stage whose observations
    # are missing, so the shadow cannot name it unreached.
    shadow = _shadow(tmp_path, _stopping_gates())
    if missing == "a stage saved nothing":
        note = "the first stage saved no observations"
        shadow.note_missing_observations(note)
    else:
        torn = tmp_path / "first.observations.json"
        torn.write_text("{")
        shadow.load_observations(torn)
        note = "first.observations.json was not read whole"
    _stop_on_a_failed_beta(shadow)
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["phases_recorded"] == []
    assert not shadow.report_path.exists()
    assert payload["observations_missing"] == [note]
    # A file that cannot be read is also a shadow error; a stage that saved
    # nothing is only a note.
    assert len(payload["shadow_errors"]) == (1 if missing == "a file is torn" else 0)


def test_a_note_of_missing_observations_alone_passes_a_finished_run(
    tmp_path,
) -> None:
    # A stage that saved nothing and observed nothing (ACS's qa stage
    # declares no entry) leaves a finished run's verdict to its entries.
    shadow = _shadow(tmp_path, _two_phase_gates())
    shadow.note_missing_observations("the qa stage saved no observations")
    with shadow:
        for entry_id, gate, key in (
            ("us_test_alpha", ALPHA, "alpha_value"),
            ("us_test_beta", BETA, "beta_value"),
            ("us_test_final", ALPHA, "alpha_value"),
        ):
            shadow.gate(
                entry_id,
                lambda gate=gate: gate(0.5),
                EvidenceContext(artifacts={key: 0.5}),
            )
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is False
    assert payload["phases_recorded"] == ["assembled", "terminal"]
    assert payload["observations_missing"] == ["the qa stage saved no observations"]


def test_the_run_refuses_a_stopped_phase_that_drops_an_entry_it_evaluated(
    tmp_path,
) -> None:
    # In one process the run's own refusal (#1609) backs the shadow up: an
    # entry evaluated through the run but never observed would otherwise be
    # named unreached. The shadow records the refusal, never raises, and still
    # lists the phase the run had already written.
    gates = _manifest(
        [
            _entry("us_test_alpha", "alpha"),
            _entry("us_test_first", "alpha", phase="terminal"),
            _entry("us_test_second", "beta", phase="terminal"),
            _entry("us_test_third", "alpha", phase="terminal"),
        ]
    )
    shadow = _shadow(tmp_path, gates)
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    # A wiring fault: the run evaluates an entry the shadow never observes.
    shadow.run.evaluate_entries(
        ("us_test_first",), EvidenceContext(artifacts={"alpha_value": 0.5})
    )
    _stop_on_a_failed_beta(shadow)
    payload = json.loads(shadow.comparison_path.read_text())
    assert payload["phases_recorded"] == ["assembled"]
    assert payload["report_file"] == shadow.report_path.name
    report = json.loads(shadow.report_path.read_text())
    assert report["blocked_at_phase"] is None
    assert report["gates"]["us_test_alpha"]["status"] == "passed"
    (error,) = payload["shadow_errors"]
    assert "marks ['us_test_first'] unreached, but this run evaluated" in error


def test_the_shadow_never_raises_while_the_tool_runs(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates)

    def broken_context():
        raise ValueError("cannot build evidence")

    assert (
        shadow.check(
            "us_undeclared", EvidenceContext(), LegacyVerdict.from_failures(())
        )
        is None
    )
    assert (
        shadow.check("us_test_alpha", broken_context, LegacyVerdict.from_failures(()))
        is None
    )
    assert shadow.check("us_test_beta", EvidenceContext(), "passed") is None
    # The same entry twice is a wiring fault, recorded once.
    shadow.gate(
        "us_test_final",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    shadow.gate(
        "us_test_final",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    differences = shadow.differences()
    assert len(differences) == 4
    assert any("us_undeclared" in line for line in differences)
    assert any("cannot build evidence" in line for line in differences)
    assert any("more than once" in line for line in differences)
    with pytest.raises(GateBatteryShadowMismatchError):
        shadow.finish()


def test_a_gate_today_did_not_run_must_be_evidence_absent(tmp_path) -> None:
    gates = _two_phase_gates()
    shadow = _shadow(tmp_path, gates)
    shadow.not_run("us_test_alpha", "pool base: the alpha stage did not run")
    shadow.gate(
        "us_test_beta",
        lambda: BETA(0.5),
        EvidenceContext(artifacts={"beta_value": 0.5}),
    )
    shadow.gate(
        "us_test_final",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    payload = shadow.finish()
    (alpha,) = [row for row in payload["entries"] if row["id"] == "us_test_alpha"]
    assert alpha["legacy"] == {
        "status": "not_run",
        "failures": [],
        "error": None,
        "reason": "pool base: the alpha stage did not run",
    }
    assert alpha["battery"]["status"] == "evidence_absent"
    assert payload["verdict_difference"] is False


def test_failure_line_gates_compare_their_lines(tmp_path) -> None:
    def spm_lines(value, *, cap=1.0):
        return [f"spm: {value} exceeds {cap}."] if value > cap else []

    specs = (
        USGateSpec(
            check="spm",
            legacy_name="us_spm_composition",
            gate="tool:spm_lines",
            positional=("value",),
            parameters=frozenset({"cap"}),
            returns="failures",
        ),
    )
    gates = _manifest(
        [_entry("us_test_spm", "spm", phase="terminal")], phases=["terminal"]
    )
    registry = us_check_binding("stage_health", specs, tool={"spm_lines": spm_lines})
    shadow = _shadow(tmp_path, gates, registry=registry)
    lines = shadow.failures(
        "us_test_spm", lambda: spm_lines(2.0), EvidenceContext(artifacts={"value": 2.0})
    )
    assert lines == ["spm: 2.0 exceeds 1.0."]
    payload = shadow.finish()
    assert payload["verdict_difference"] is False
    assert payload["entries"][0]["battery"]["failures"] == lines


def test_stale_shadow_files_are_removed_when_a_run_starts(tmp_path) -> None:
    report, comparison = shadow_file_paths(tmp_path, "us_test")
    report.write_text("{}")
    comparison.write_text("{}")
    _shadow(tmp_path, _two_phase_gates())
    assert not report.exists() and not comparison.exists()


def _earlier_files(tmp_path):
    report, comparison = shadow_file_paths(tmp_path, "us_test")
    report.write_text('{"earlier": "report"}')
    comparison.write_text('{"earlier": "comparison"}')
    return report, comparison


def test_held_files_survive_a_run_that_stops_before_it_lets_them_go(
    tmp_path,
) -> None:
    report, comparison = _earlier_files(tmp_path)
    shadow = _shadow(tmp_path, _two_phase_gates(), hold_files=True)
    shadow.gate(
        "us_test_alpha",
        lambda: ALPHA(0.5),
        EvidenceContext(artifacts={"alpha_value": 0.5}),
    )
    shadow.stop(RuntimeError("refused before the first side effect"))
    assert json.loads(report.read_text()) == {"earlier": "report"}
    assert json.loads(comparison.read_text()) == {"earlier": "comparison"}


def test_released_files_are_cleared_and_then_written(tmp_path) -> None:
    report, comparison = _earlier_files(tmp_path)
    shadow = _shadow(tmp_path, _two_phase_gates(), hold_files=True)
    shadow.release_files()
    assert not report.exists() and not comparison.exists()
    shadow.release_files()  # once is enough; a second call changes nothing
    shadow.stop(RuntimeError("a later failure"))
    payload = json.loads(comparison.read_text())
    assert payload["stopped_by"]["type"] == "RuntimeError"


def test_a_finished_run_writes_its_files_even_while_held(tmp_path) -> None:
    _, comparison = _earlier_files(tmp_path)
    gates = _two_phase_gates()
    with _shadow(tmp_path, gates, hold_files=True) as shadow:
        for entry_id, value in (
            ("us_test_alpha", 0.5),
            ("us_test_beta", 2.0),
            ("us_test_final", 0.1),
        ):
            gate = BETA if entry_id == "us_test_beta" else ALPHA
            key = "beta_value" if entry_id == "us_test_beta" else "alpha_value"
            shadow.gate(
                entry_id,
                lambda gate=gate, value=value: gate(value),
                EvidenceContext(artifacts={key: value}),
            )
    assert json.loads(comparison.read_text())["run_completed"] is True


def test_the_suite_runs_the_tools_without_their_shadows(monkeypatch) -> None:
    # tests/conftest.py spells the switch out; this holds it to the constant.
    assert os.environ.get(US_GATE_SHADOW_SWITCH) == "off"
    assert tool_gate_shadows_enabled() is False
    for value in (" OFF ", "Off"):
        monkeypatch.setenv(US_GATE_SHADOW_SWITCH, value)
        assert tool_gate_shadows_enabled() is False
    for value in ("", "on", "0", "false"):
        monkeypatch.setenv(US_GATE_SHADOW_SWITCH, value)
        assert tool_gate_shadows_enabled() is True
    monkeypatch.delenv(US_GATE_SHADOW_SWITCH)
    assert tool_gate_shadows_enabled() is True


def test_waivers_and_overrides_must_name_scope_entries(tmp_path) -> None:
    with pytest.raises(ValueError, match="outside scope"):
        _shadow(tmp_path, _two_phase_gates(), waivers={"us_elsewhere": ["--allow-x"]})
    with pytest.raises(ValueError, match="outside scope"):
        _shadow(
            tmp_path,
            _two_phase_gates(),
            overrides={"us_elsewhere": {"cap": {"declared": 1, "runtime": 2}}},
        )


def test_shadow_file_names_follow_the_scope() -> None:
    report, comparison = shadow_file_paths("/out/gate_battery_shadow", "us_national")
    assert report.name == "us_national.gate_battery_shadow_report.json"
    assert comparison.name == "us_national.gate_battery_shadow_comparison.json"
    with pytest.raises(ValueError):
        shadow_file_paths("/out", "../escape")


# ---------------------------------------------------------------------------
# The US check binding
# ---------------------------------------------------------------------------


def test_the_check_binding_routes_evidence_and_thresholds() -> None:
    binding = _registry()["stage_health"]
    assert binding.parameter_keys == frozenset({"check", "cap"})
    alpha = {"check": "alpha", "cap": 0.25}
    assert binding.required_artifacts(alpha) == frozenset({"alpha_value"})
    assert binding.requires_frame(alpha) is False
    result = binding.evaluate(EvidenceContext(artifacts={"alpha_value": 0.5}), alpha)
    # The declared threshold, not the function's default, decides; the
    # legacy name is re-minted under the declared gate.
    assert result.name == "stage_health"
    assert result.failures == ("us_alpha_signal: 0.5 exceeds 0.25.",)
    beta = binding.evaluate(
        EvidenceContext(artifacts={"beta_value": 0.5}), {"check": "beta"}
    )
    assert beta.passed and beta.name == "stage_health"


def test_the_check_binding_fails_closed() -> None:
    binding = _registry()["stage_health"]
    with pytest.raises(ValueError, match="no US gate is bound"):
        binding.evaluate(EvidenceContext(), {"check": "gamma"})
    assert binding.required_artifacts({"check": "gamma"}) == frozenset()
    spec = USGateSpec(
        check="alpha",
        legacy_name="us_alpha_signal",
        gate="tool:ALPHA",
        positional=("alpha_value",),
    )
    strict = USCheckBinding("stage_health", (spec,), tool=TOOL)
    with pytest.raises(ValueError, match="does not take parameter"):
        strict.evaluate(
            EvidenceContext(artifacts={"alpha_value": 0.5}),
            {"check": "alpha", "cap": 1.0},
        )
    # Another name passes through, so the executor's name check fails it.
    other = USCheckBinding(
        "stage_health",
        (
            USGateSpec(
                check="alpha",
                legacy_name="us_something_else",
                gate="tool:ALPHA",
                positional=("alpha_value",),
            ),
        ),
        tool=TOOL,
    )
    result = other.evaluate(
        EvidenceContext(artifacts={"alpha_value": 0.5}), {"check": "alpha"}
    )
    assert result.name == "us_alpha_signal"


def test_pinned_thresholds_must_equal_todays_constants() -> None:
    namespace = {"GATE": ALPHA, "_CAP": 1.0}
    spec = USGateSpec(
        check="alpha",
        legacy_name="us_alpha_signal",
        gate="tool:GATE",
        positional=("alpha_value",),
        pinned={"cap": "tool:_CAP"},
    )
    binding = USCheckBinding("stage_health", (spec,), tool=namespace)
    assert binding.parameter_keys == frozenset({"check", "cap"})
    assert binding.pinned_values("alpha") == {"cap": 1.0}
    evidence = EvidenceContext(artifacts={"alpha_value": 1.5})
    # The declared value equals today's constant: today's gate decides with it.
    assert binding.evaluate(evidence, {"check": "alpha", "cap": 1.0}).failures == (
        "us_alpha_signal: 1.5 exceeds 1.0.",
    )
    with pytest.raises(ValueError, match="differs from today's"):
        binding.evaluate(evidence, {"check": "alpha", "cap": 2.0})
    with pytest.raises(ValueError, match="does not declare"):
        binding.evaluate(evidence, {"check": "alpha"})
    # A later patch of the constant reaches the next evaluation.
    namespace["_CAP"] = 2.0
    with pytest.raises(ValueError, match="differs from today's"):
        binding.evaluate(evidence, {"check": "alpha", "cap": 1.0})


def test_routed_bands_reach_the_gate_as_tuples() -> None:
    seen = []

    def gate(value, *, bounds=(0.0, 1.0)):
        seen.append(bounds)
        return GateResult(name="us_band_signal", passed=True)

    spec = USGateSpec(
        check="band",
        legacy_name="us_band_signal",
        gate="tool:gate",
        positional=("value",),
        parameters=frozenset({"bounds"}),
    )
    binding = USCheckBinding("stage_health", (spec,), tool={"gate": gate})
    binding.evaluate(
        EvidenceContext(artifacts={"value": 0.5}), {"check": "band", "bounds": [0, 1]}
    )
    assert seen == [(0, 1)]


def test_tool_references_are_resolved_up_front() -> None:
    spec = USGateSpec(
        check="tool",
        legacy_name="us_tool_signal",
        gate="tool:_tool_gate",
        arguments={"frame": FRAME},
    )
    assert spec.needs_frame and spec.references == ("tool:_tool_gate",)
    with pytest.raises(ValueError, match="no tool was given"):
        USCheckBinding("stage_health", (spec,))
    with pytest.raises(ValueError, match="no global '_tool_gate'"):
        USCheckBinding("stage_health", (spec,), tool={"_other": BETA})
    with pytest.raises(ValueError, match="is not callable"):
        USCheckBinding("stage_health", (spec,), tool={"_tool_gate": 3})
    missing_module = USGateSpec(
        check="tool",
        legacy_name="us_tool_signal",
        gate="microcosm.build.no_such_module:gate",
    )
    with pytest.raises(ValueError, match="unresolved references"):
        USCheckBinding("stage_health", (missing_module,), tool={})
    namespace = {
        "_tool_gate": lambda frame: GateResult(name="us_tool_signal", passed=True)
    }
    binding = USCheckBinding("stage_health", (spec,), tool=namespace)
    assert binding.requires_frame({"check": "tool"})
    assert binding.evaluate(EvidenceContext(frame=object()), {"check": "tool"}).passed
    # The binding calls what the namespace holds when it runs.
    namespace["_tool_gate"] = lambda frame: GateResult(
        name="us_tool_signal", passed=False, failures=("patched",)
    )
    assert binding.evaluate(
        EvidenceContext(frame=object()), {"check": "tool"}
    ).failures == ("patched",)


def test_a_gate_spec_refuses_ambiguous_declarations() -> None:
    with pytest.raises(ValueError, match="dispatch key"):
        USGateSpec(
            check="a", legacy_name="x", gate="tool:A", parameters=frozenset({"check"})
        )
    with pytest.raises(ValueError, match="both evidence and thresholds"):
        USGateSpec(
            check="a",
            legacy_name="x",
            gate="tool:A",
            arguments={"cap": "cap_value"},
            parameters=frozenset({"cap"}),
        )
    with pytest.raises(ValueError, match="both routed and pinned"):
        USGateSpec(
            check="a",
            legacy_name="x",
            gate="tool:A",
            parameters=frozenset({"cap"}),
            pinned={"cap": "tool:_CAP"},
        )
    with pytest.raises(ValueError, match="must be 'tool:<name>'"):
        USGateSpec(check="a", legacy_name="x", gate="no_separator")
    with pytest.raises(ValueError, match="declared twice"):
        USCheckBinding(
            "stage_health",
            (
                USGateSpec(check="a", legacy_name="x", gate="tool:ALPHA"),
                USGateSpec(check="a", legacy_name="y", gate="tool:BETA"),
            ),
            tool=TOOL,
        )


def test_the_registry_is_fresh_on_every_call() -> None:
    first = _registry()
    second = _registry()
    assert first is not second
    assert first["stage_health"] is not second["stage_health"]
    # The shared registry is never mutated.
    assert "stage_health" not in DEFAULT_REGISTRY
