"""ACS local runs the gate battery in shadow, stage by stage (#1585 item 11).

``tools/build_us_acs_local_release.py`` may run each ``--stage`` as its own
process, so every stage saves what it observed beside the checkpoint and the
scope's report and comparison are rebuilt from every stage saved so far.
Only a finished package stage decides. These tests drive the stage seam with
stand-in gates; no stage runs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.gate_scopes import us_scoped_gate_manifest
from microcosm.build.us_runtime.gate_shadow import (
    US_GATE_SHADOW_SWITCH,
    GateBatteryShadowMismatchError,
    current_gate_shadow,
    shadow_file_paths,
    shadow_not_run,
)
from test_support.microcosm_build.us_gate_shadow import (
    arm_us_tool_gate_shadows,
    load_us_gate_tool,
    observe_entries,
    stand_in_gates,
)


@pytest.fixture(autouse=True)
def _tool_shadows_on(monkeypatch) -> None:
    arm_us_tool_gate_shadows(monkeypatch)


#: Which stage evaluates each entry (the qa stage has none).
STAGE_ENTRIES = {
    "materialize": (
        "us_local_hours_materialize",
        "us_local_self_employment_identity_materialize",
        "us_local_ssi_medicaid_take_up_materialize",
    ),
    "calibrate": (
        "us_local_hours_calibrate",
        "us_local_self_employment_identity_calibrate",
    ),
    "qa": (),
    "finalize": (
        "us_local_hours_finalize",
        "us_local_puma_ladder",
        "us_local_hours_worked",
        "us_local_take_up",
        "us_local_immigration",
        "us_local_work_disability",
        "us_local_income_transfer",
        "us_local_ssi_disability",
        "us_local_ssi_medicaid_take_up_finalize",
        "us_local_receipt_anchors",
        "us_local_reviewed_fill_consumers_finalize",
        "us_local_district_ess_collapse",
    ),
    "package": (
        "us_local_stored_inputs",
        "us_local_work_experience_inputs",
        "us_local_spm_measurement",
        "us_local_hours_package",
        "us_local_self_employment_identity_package",
        "us_local_reviewed_fill_consumers_package",
    ),
}


@pytest.fixture
def acs():
    return load_us_gate_tool("us_local")


def _args(tmp_path):
    return SimpleNamespace(
        checkpoint_dir=tmp_path / "checkpoints",
        district_ess_relative_floor=0.25,
        district_ess_floor=15.0,
    )


def _run_stage(acs, args, stage, *, legacy=None):
    with acs._local_gate_shadow_stage(args, stage):
        observe_entries("us_local", acs, STAGE_ENTRIES[stage], legacy=legacy)
        if stage == "finalize":
            shadow_not_run(
                "us_local_block_location",
                "the staging used the legacy location rule; no block ladder",
            )


def _comparison(args):
    _, comparison = shadow_file_paths(acs_directory(args), "us_local")
    return json.loads(comparison.read_text())


def acs_directory(args):
    return args.checkpoint_dir / "gate_battery_shadow"


def test_the_stage_map_covers_the_scope() -> None:
    covered = {entry_id for ids in STAGE_ENTRIES.values() for entry_id in ids}
    covered.add("us_local_block_location")
    assert covered == {entry.id for entry in us_scoped_gate_manifest("us_local").gates}


def test_separate_stage_processes_add_up_to_one_finished_run(
    tmp_path, monkeypatch, acs
) -> None:
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    _run_stage(acs, args, "materialize")
    after_materialize = _comparison(args)
    assert after_materialize["run_completed"] is False
    assert after_materialize["stopped_at"] == "after the materialize stage"
    for stage in ("calibrate", "qa", "finalize", "package"):
        _run_stage(acs, args, stage)
    payload = _comparison(args)
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is False
    assert payload["phases_recorded"] == ["assembled", "terminal"]
    for stage in STAGE_ENTRIES:
        assert (acs_directory(args) / f"{stage}.observations.json").is_file()


def test_with_the_switch_off_a_stage_touches_no_shadow_file(
    tmp_path, monkeypatch, acs
) -> None:
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    _run_stage(acs, args, "materialize")
    saved = acs_directory(args) / "materialize.observations.json"
    before = saved.read_bytes()
    monkeypatch.setenv(US_GATE_SHADOW_SWITCH, "off")
    with acs._local_gate_shadow_stage(args, "materialize") as shadow:
        assert shadow is None
        assert current_gate_shadow() is None
    # A rerun with the switch off neither discards nor rewrites what was saved.
    assert saved.read_bytes() == before


def test_a_finalize_stop_is_recorded_and_its_rerun_starts_clean(
    tmp_path, monkeypatch, acs
) -> None:
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs, failing={"take_up_signal"})
    for stage in ("materialize", "calibrate", "qa"):
        _run_stage(acs, args, stage)
    with pytest.raises(SystemExit, match="hard gate failure"):
        with acs._local_gate_shadow_stage(args, "finalize"):
            observe_entries("us_local", acs, STAGE_ENTRIES["finalize"])
            raise SystemExit("finalize: hard gate failure(s): take-up")
    payload = _comparison(args)
    assert payload["stopped_by"]["type"] == "SystemExit"
    (take_up,) = [row for row in payload["entries"] if row["id"] == "us_local_take_up"]
    assert take_up["match"] is True and take_up["battery"]["status"] == "failed"
    # Rerunning an earlier stage discards its own and every later stage's file.
    (acs_directory(args) / "package.observations.json").write_text("{}")
    _run_stage(acs, args, "calibrate")
    assert not (acs_directory(args) / "finalize.observations.json").exists()
    assert not (acs_directory(args) / "package.observations.json").exists()


def test_a_difference_fails_the_finished_package_stage(
    tmp_path, monkeypatch, acs
) -> None:
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs, failing={"local_hours_signal"})
    for stage in ("materialize", "calibrate", "qa"):
        _run_stage(acs, args, stage)
    passing = GateResult(name="acs_local_hours_signal", passed=True)
    legacy = {
        entry_id: (lambda: passing)
        for entry_id in ("us_local_hours_finalize", "us_local_hours_package")
    }
    # Earlier stages record and go on, whatever they saw.
    _run_stage(acs, args, "finalize", legacy=legacy)
    with pytest.raises(GateBatteryShadowMismatchError, match="us_local_hours"):
        _run_stage(acs, args, "package", legacy=legacy)
