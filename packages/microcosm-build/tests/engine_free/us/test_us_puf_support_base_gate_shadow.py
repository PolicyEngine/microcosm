"""Route A's base builder runs the gate battery in shadow (microcosm#1585 item 11).

``tools/build_us_puf_support_base.py`` arms the ``us_base`` scope around its
single-process pipeline, and around each checkpointed stage, whose child
process saves what it observed for the parent to finish. These tests drive
those seams with stand-in gates; the stages themselves are not run.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from microcosm.build import FitWeightRecord, GateResult
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.us_runtime.battery_bindings import us_gate_specs
from microcosm.build.us_runtime.gate_scopes import us_scoped_gate_manifest
from microcosm.build.us_runtime.gate_shadow import (
    GATE_BATTERY_SHADOW_DIRECTORY,
    US_GATE_SHADOW_SWITCH,
    active_gate_shadow,
    current_gate_shadow,
    shadow_file_paths,
    shadow_gate,
)
from test_support.microcosm_build.us_gate_shadow import (
    arm_us_tool_gate_shadows,
    load_us_gate_tool,
)


@pytest.fixture(autouse=True)
def _tool_shadows_on(monkeypatch) -> None:
    arm_us_tool_gate_shadows(monkeypatch)


@pytest.fixture
def base():
    return load_us_gate_tool("us_base")


def _args(tmp_path, **overrides):
    values = {
        "target_year": 2024,
        "out": tmp_path / "out",
        "checkpoint_dir": tmp_path / "checkpoints",
        "stage": "all",
        "allow_geography_ladder_gate_failures": False,
        "block_ladder_artifact": None,
        "equivalence_boundary_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _stand_in_gates(monkeypatch, base, *, failing=()):
    """Replace every base gate the bindings call with a stand-in verdict."""

    for spec in us_gate_specs("us_base"):
        name = spec.gate.removeprefix("tool:")
        failed = spec.check in failing

        def gate(*args, _legacy=spec.legacy_name, _failed=failed, **kwargs):
            if _failed:
                return GateResult(
                    name=_legacy, passed=False, failures=(f"{_legacy}: stand-in.",)
                )
            return GateResult(name=_legacy, passed=True)

        monkeypatch.setitem(vars(base), name, gate)


def _observe_every_entry(base, args):
    """What a passing pipeline observes: every frame gate, the audit, location."""

    tool = vars(base)
    frame = object()
    specs = {spec.check: spec for spec in us_gate_specs("us_base")}
    for entry in us_scoped_gate_manifest("us_base").gates:
        if entry.gate == "weights_audit":
            records = [FitWeightRecord("us_puf_tax_detail_support", "design")]
            shadow_gate(
                entry.id,
                lambda records=records: tool["weights_audit_gate"](records),
                EvidenceContext(artifacts={"fit_weight_records": records}),
            )
        elif entry.parameters["check"] in {"geography_ladder", "block_location"}:
            continue
        else:
            gate = tool[specs[entry.parameters["check"]].gate.removeprefix("tool:")]
            base._shadow_frame_gate(entry.id, lambda gate=gate: gate(frame), frame)
    base._shadow_legacy_location_not_run(args)


def test_the_single_process_run_finishes_its_shadow(tmp_path, monkeypatch, base):
    args = _args(tmp_path, checkpoint_dir=None)
    _stand_in_gates(monkeypatch, base)
    monkeypatch.setattr(base, "_verify_asec_source_digests", lambda args: None)
    monkeypatch.setattr(base, "_run_all", lambda args: _observe_every_entry(base, args))
    base._dispatch(args, None)
    _, comparison = shadow_file_paths(
        args.out.resolve() / GATE_BATTERY_SHADOW_DIRECTORY, "us_base"
    )
    payload = json.loads(comparison.read_text())
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is False
    assert payload["phases_recorded"] == ["assembled", "transferred", "terminal"]
    location = {
        row["id"]: row
        for row in payload["entries"]
        if row["id"] in {"us_base_geography_ladder", "us_base_block_location"}
    }
    assert {row["legacy"]["status"] for row in location.values()} == {"not_run"}
    assert {row["battery"]["status"] for row in location.values()} == {
        "evidence_absent"
    }


def test_with_the_switch_off_no_shadow_is_armed_or_written(tmp_path, monkeypatch, base):
    monkeypatch.setenv(US_GATE_SHADOW_SWITCH, "off")
    args = _args(tmp_path, checkpoint_dir=None)
    monkeypatch.setattr(base, "_verify_asec_source_digests", lambda args: None)
    armed = []
    monkeypatch.setattr(
        base, "_run_all", lambda args: armed.append(current_gate_shadow())
    )
    base._dispatch(args, None)
    assert armed == [None]
    with base._armed_stage_gate_shadow(_args(tmp_path, stage="wic_post_clone")) as save:
        assert current_gate_shadow() is None
        save()
    assert not (args.out / GATE_BATTERY_SHADOW_DIRECTORY).exists()
    assert not (tmp_path / "checkpoints" / GATE_BATTERY_SHADOW_DIRECTORY).exists()


def test_a_verdict_difference_is_recorded_without_failing_the_single_process_run(
    tmp_path, monkeypatch, base, capsys
):
    args = _args(tmp_path, checkpoint_dir=None)
    # The battery's gate fails where today's verdict passed.
    _stand_in_gates(monkeypatch, base, failing={"pregnancy_signal"})
    monkeypatch.setattr(base, "_verify_asec_source_digests", lambda args: None)

    def run_all(args):
        _observe_every_entry(base, args)

    original = base._shadow_frame_gate

    def legacy_passes(entry_id, evaluate, frame):
        if entry_id == "us_base_pregnancy":
            return original(
                entry_id,
                lambda: GateResult(name="pregnancy_signal", passed=True),
                frame,
            )
        return original(entry_id, evaluate, frame)

    monkeypatch.setattr(base, "_shadow_frame_gate", legacy_passes)
    monkeypatch.setattr(base, "_run_all", run_all)
    # Today's gates still decide: the run returns, with the difference on record.
    base._dispatch(args, None)
    _, comparison = shadow_file_paths(
        args.out.resolve() / GATE_BATTERY_SHADOW_DIRECTORY, "us_base"
    )
    payload = json.loads(comparison.read_text())
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is True
    assert [row["id"] for row in payload["differences"]] == ["us_base_pregnancy"]
    assert "scope 'us_base' disagrees with today's gates" in capsys.readouterr().err


def test_todays_raise_propagates_and_the_shadow_says_where(tmp_path, monkeypatch, base):
    args = _args(tmp_path, checkpoint_dir=None)
    _stand_in_gates(monkeypatch, base, failing={"relationship_inputs_signal"})
    monkeypatch.setattr(base, "_verify_asec_source_digests", lambda args: None)
    tool = vars(base)

    def run_all(args):
        frame = object()
        gate = base._shadow_frame_gate(
            "us_base_relationship_inputs",
            lambda: tool["us_relationship_inputs_signal_gate"](frame),
            frame,
        )
        if not gate.passed:
            raise SystemExit("Relationship-input signal gate failed")

    monkeypatch.setattr(base, "_run_all", run_all)
    with pytest.raises(SystemExit, match="Relationship-input signal gate failed"):
        base._dispatch(args, None)
    report, comparison = shadow_file_paths(
        args.out.resolve() / GATE_BATTERY_SHADOW_DIRECTORY, "us_base"
    )
    payload = json.loads(comparison.read_text())
    assert payload["run_completed"] is False
    assert payload["stopped_by"]["type"] == "SystemExit"
    assert payload["blocked_at_phase"] == "assembled"
    gates = json.loads(report.read_text())["gates"]
    assert gates["us_base_relationship_inputs"]["status"] == "failed"
    assert gates["us_base_pregnancy"]["status"] == "unreached"


def test_checkpointed_stages_carry_their_observations_to_the_parent(
    tmp_path, monkeypatch, base
):
    _stand_in_gates(monkeypatch, base, failing={"medicare_take_up_input_signal"})
    tool = vars(base)
    frame = object()
    wic = _args(tmp_path, stage="wic_post_clone")
    with base._armed_stage_gate_shadow(wic) as save:
        base._shadow_frame_gate(
            "us_base_wic_claim_post_clone",
            lambda: tool["us_wic_claim_signal_gate"](frame),
            frame,
        )
        save()
    assert base._stage_gate_shadow_observations(wic, "wic_post_clone").is_file()

    medicare = _args(tmp_path, stage="medicare_post_clone")
    with pytest.raises(SystemExit):
        with base._armed_stage_gate_shadow(medicare):
            gate = base._shadow_frame_gate(
                "us_base_medicare_take_up_post_clone",
                lambda: tool["us_medicare_take_up_signal_gate"](frame),
                frame,
            )
            if not gate.passed:
                raise SystemExit("Medicare take-up input signal gate failed")
    # The stage that raised still saved what it saw.
    assert base._stage_gate_shadow_observations(
        medicare, "medicare_post_clone"
    ).is_file()

    parent = _args(tmp_path)
    shadow = base._base_gate_shadow(parent, base._final_gate_shadow_directory(parent))
    base._load_stage_gate_shadows(
        shadow, parent, ("wic_post_clone", "medicare_post_clone")
    )
    shadow.stop(SystemExit("Checkpointed stage 'medicare_post_clone' failed"))
    payload = json.loads(shadow.comparison_path.read_text())
    observed = {row["id"]: row for row in payload["entries"]}
    assert set(observed) == {
        "us_base_wic_claim_post_clone",
        "us_base_medicare_take_up_post_clone",
    }
    assert all(row["match"] for row in observed.values())
    assert observed["us_base_medicare_take_up_post_clone"]["battery"]["status"] == (
        "failed"
    )
    assert payload["shadow_errors"] == []
    assert payload["observations_missing"] == []


def test_the_parent_notes_each_stage_that_ran_and_saved_nothing(
    tmp_path, monkeypatch, base
):
    # A stage that ran but left no file (its save failed, or it ran with the
    # shadow off) may have reached entries the parent then lacks, so it is
    # noted and a stopped run never names them unreached. The stage after the
    # one that failed never ran, so it is not.
    steps = ("source_construction", "pre_clone_enrichment", "clone_feature_extraction")

    class Runtime:
        def __init__(self, *_args, **_kwargs) -> None:
            self.context = SimpleNamespace(completed=steps[:1])

    monkeypatch.setattr(base, "_pipeline_steps_for", lambda args: steps)
    monkeypatch.setattr(base, "_pipeline_for", lambda args: None)
    monkeypatch.setattr(base, "StageRuntime", Runtime)
    monkeypatch.setattr(base, "_stage_run_config", lambda args: {})
    monkeypatch.setattr(base, "_stage_cli_args", lambda args, stage: ["--stage", stage])
    monkeypatch.setattr(
        base.subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=1)
    )
    args = _args(tmp_path)
    with pytest.raises(SystemExit, match="'pre_clone_enrichment' failed"):
        base._run_staged_all(args)
    _, comparison = shadow_file_paths(
        base._final_gate_shadow_directory(args), "us_base"
    )
    payload = json.loads(comparison.read_text())
    assert payload["run_completed"] is False
    assert payload["observations_missing"] == [
        "the source_construction stage saved no observations",
        "the pre_clone_enrichment stage saved no observations",
    ]
    assert payload["phases_recorded"] == []


def test_a_resume_cannot_mix_stages_run_with_and_without_the_shadow(
    tmp_path, monkeypatch, base
) -> None:
    # A stage run with the switch off saves nothing, so the parent could only
    # call its entries unwired. The run config locks the switch when it is
    # off; with the shadow on, as in every real build, it is unchanged.
    puf = tmp_path / "puf.h5"
    puf.write_bytes(b"puf")
    args = base._parse_args(
        [
            "--base-h5",
            str(tmp_path / "base.h5"),
            "--puf-h5",
            str(puf),
            "--out",
            str(tmp_path / "out"),
            "--without-block-ladder",
        ]
    )
    monkeypatch.setattr(
        base, "_builder_code_identity", lambda: {"source_sha256": "builder"}
    )
    armed = base._stage_run_config(args)
    assert "gate_battery_shadow" not in armed
    monkeypatch.setenv(US_GATE_SHADOW_SWITCH, "off")
    off = base._stage_run_config(args)
    assert off == {**armed, "gate_battery_shadow": "off"}
    checkpoints = tmp_path / "checkpoints"
    base.StageRuntime(checkpoints, base.OUTER_STAGE_PIPELINE, run_config=off)
    monkeypatch.delenv(US_GATE_SHADOW_SWITCH)
    with pytest.raises(ValueError, match="run_config differs"):
        base.StageRuntime(
            checkpoints,
            base.OUTER_STAGE_PIPELINE,
            run_config=base._stage_run_config(args),
        )


def test_the_geography_waiver_is_recorded_by_entry(tmp_path, base) -> None:
    shadow = base._base_gate_shadow(
        _args(tmp_path, allow_geography_ladder_gate_failures=True), tmp_path
    )
    payload = shadow.finish(stopped_at="test")
    assert payload["waivers"] == {
        "us_base_block_location": ["--allow-geography-ladder-gate-failures"],
        "us_base_geography_ladder": ["--allow-geography-ladder-gate-failures"],
    }


def test_the_legacy_location_rule_records_what_it_never_evaluates(
    tmp_path, base
) -> None:
    with_ladder = _args(tmp_path, block_ladder_artifact=tmp_path / "ladder.parquet")
    shadow = base._base_gate_shadow(with_ladder, tmp_path)
    with active_gate_shadow(shadow):
        base._shadow_legacy_location_not_run(with_ladder)
    # With a ladder artifact the ladder gate runs; only the block draw is absent.
    assert shadow.outcome("us_base_block_location") is not None
    assert shadow.outcome("us_base_geography_ladder") is None
