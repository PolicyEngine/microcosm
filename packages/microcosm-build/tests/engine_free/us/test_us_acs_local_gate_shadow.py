"""ACS local runs the gate battery in shadow, stage by stage (#1585 item 11).

``tools/build_us_acs_local_release.py`` may run each ``--stage`` as its own
process, so every stage saves what it observed beside the checkpoint and the
scope's report and comparison are rebuilt from every stage saved so far.
Only a finished package stage decides. These tests drive the stage seam with
stand-in gates; no stage runs.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.gate_scopes import us_scoped_gate_manifest
from microcosm.build.us_runtime.gate_shadow import (
    US_GATE_SHADOW_SWITCH,
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
        "us_local_esi_premiums_signal",
        "us_local_esi_premiums_anchor",
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


def _run_stage(acs, args, stage, *, legacy=None, report=None):
    """Run ``stage``'s stand-in gates; finalize also writes ``report``, if
    given, to ``args.gate_report``, as ``do_finalize`` writes its report."""

    with acs._local_gate_shadow_stage(args, stage):
        observe_entries("us_local", acs, STAGE_ENTRIES[stage], legacy=legacy)
        if stage == "finalize":
            shadow_not_run(
                "us_local_block_location",
                "the staging used the legacy location rule; no block ladder",
            )
            if report is not None:
                args.gate_report.write_text(json.dumps(report))


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


def test_the_esi_premium_waiver_is_recorded_on_its_two_entries(
    tmp_path, monkeypatch, acs
) -> None:
    args = _args(tmp_path)
    assert acs._local_gate_waivers(args) == {}
    args.allow_esi_premium_gaps = True
    waived = {
        "us_local_esi_premiums_signal": ["--allow-esi-premium-gaps"],
        "us_local_esi_premiums_anchor": ["--allow-esi-premium-gaps"],
    }
    assert acs._local_gate_waivers(args) == waived
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in STAGE_ENTRIES:
        _run_stage(acs, args, stage)
    payload = _comparison(args)
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is False
    assert payload["waivers"] == waived


@pytest.mark.parametrize(
    ("finalize_waived", "package_waived"), [(False, True), (True, False)]
)
def test_an_esi_entry_keeps_the_waiver_finalize_ran_with(
    tmp_path, monkeypatch, acs, finalize_waived, package_waived
) -> None:
    # The stages may run as separate processes with different flags. The ESI
    # entries are evaluated at finalize, so the comparison the package stage
    # rebuilds records finalize's waiver on them, never the package stage's.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in STAGE_ENTRIES:
        args.allow_esi_premium_gaps = (
            package_waived if stage == "package" else finalize_waived
        )
        _run_stage(acs, args, stage)

    payload = _comparison(args)
    flags = ["--allow-esi-premium-gaps"] if finalize_waived else []
    waived_by = {row["id"]: row["waived_by"] for row in payload["entries"]}
    assert payload["run_completed"] is True
    assert waived_by["us_local_esi_premiums_signal"] == flags
    assert waived_by["us_local_esi_premiums_anchor"] == flags
    assert payload["waivers"] == (
        {
            "us_local_esi_premiums_signal": flags,
            "us_local_esi_premiums_anchor": flags,
        }
        if finalize_waived
        else {}
    )
    # Rerunning an earlier stage discards finalize's record with its
    # observations.
    record = acs_directory(args) / "finalize.waivers.json"
    assert record.is_file()
    _run_stage(acs, args, "calibrate")
    assert not record.exists()


def test_a_finalize_checkpoint_without_a_waiver_record_marks_the_waiver_unknown(
    tmp_path, monkeypatch, acs
) -> None:
    # A finalize checkpoint saved before finalize kept a waiver record has
    # its observations and no record. Package cannot tell which waiver that
    # finalize ran with, so the ESI entries record it as unknown, not as
    # none, and the comparison is still rebuilt.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in ("materialize", "calibrate", "qa", "finalize"):
        _run_stage(acs, args, stage)
    (acs_directory(args) / "finalize.waivers.json").unlink()
    assert (acs_directory(args) / "finalize.observations.json").is_file()
    _run_stage(acs, args, "package")

    payload = _comparison(args)
    unknown = ["unknown: the finalize stage saved no waiver record"]
    esi = {"us_local_esi_premiums_signal", "us_local_esi_premiums_anchor"}
    waived_by = {row["id"]: row["waived_by"] for row in payload["entries"]}
    assert payload["run_completed"] is True
    assert {entry_id: waived_by[entry_id] for entry_id in esi} == dict.fromkeys(
        esi, unknown
    )
    assert all(
        flags == [] for entry_id, flags in waived_by.items() if entry_id not in esi
    )
    assert payload["waivers"] == dict.fromkeys(esi, unknown)


#: The entries finalize evaluates that --allow-esi-premium-gaps waives.
ESI = ("us_local_esi_premiums_signal", "us_local_esi_premiums_anchor")
WAIVED = ["--allow-esi-premium-gaps"]


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _esi_waived_by(args) -> dict[str, list[str]]:
    waived_by = {row["id"]: row["waived_by"] for row in _comparison(args)["entries"]}
    return {entry_id: waived_by[entry_id] for entry_id in ESI}


def _finalize_record(args) -> dict:
    return json.loads((acs_directory(args) / "finalize.waivers.json").read_text())


def test_package_shows_a_finalize_waiver_only_on_the_gate_report_that_run_wrote(
    tmp_path, monkeypatch, acs
) -> None:
    # Finalize a.json without the waiver, then b.json with it, on one
    # checkpoint. Package a.json ships a report that enforced the ESI gates,
    # so its shadow must not show b's waiver: the record b left names b.json.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in ("materialize", "calibrate", "qa"):
        _run_stage(acs, args, stage)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    args.gate_report, args.allow_esi_premium_gaps = a, False
    _run_stage(acs, args, "finalize", report={"run": "a", "enforced": True})
    args.gate_report, args.allow_esi_premium_gaps = b, True
    _run_stage(acs, args, "finalize", report={"run": "b", "enforced": False})

    args.gate_report, args.allow_esi_premium_gaps = a, False
    _run_stage(acs, args, "package")
    mismatch = [
        "unknown: the finalize stage's waiver record belongs to gate report "
        f"{_sha256(b)}, not to the selected gate report {_sha256(a)}"
    ]
    assert _comparison(args)["run_completed"] is True
    assert _esi_waived_by(args) == dict.fromkeys(ESI, mismatch)
    assert _comparison(args)["waivers"] == dict.fromkeys(ESI, mismatch)

    # On the report that run wrote, its waiver, whatever package's flag says.
    args.gate_report = b
    _run_stage(acs, args, "package")
    assert _esi_waived_by(args) == dict.fromkeys(ESI, WAIVED)
    assert _finalize_record(args) == {
        "gate_report_sha256": _sha256(b),
        "waivers": dict.fromkeys(ESI, WAIVED),
    }


def test_a_finalize_run_that_wrote_no_gate_report_names_none(
    tmp_path, monkeypatch, acs
) -> None:
    # A finalize run stopped before it wrote its report leaves the previous
    # run's report in place. Its waiver record names no report, so package on
    # that earlier report does not show the stopped run's waiver.
    args = _args(tmp_path)
    args.gate_report = tmp_path / "gate_summary.json"
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in ("materialize", "calibrate", "qa"):
        _run_stage(acs, args, stage)
    args.allow_esi_premium_gaps = False
    _run_stage(acs, args, "finalize", report={"run": "first"})
    earlier = _sha256(args.gate_report)
    args.allow_esi_premium_gaps = True
    with pytest.raises(SystemExit, match="stopped"):
        with acs._local_gate_shadow_stage(args, "finalize"):
            observe_entries("us_local", acs, STAGE_ENTRIES["finalize"])
            raise SystemExit("finalize stopped before it wrote its gate report")
    assert _sha256(args.gate_report) == earlier

    args.allow_esi_premium_gaps = False
    _run_stage(acs, args, "package")
    expected = [
        "unknown: the finalize stage's waiver record belongs to gate report "
        "none (that run wrote no gate report), not to the selected gate report "
        f"{earlier}"
    ]
    assert _esi_waived_by(args) == dict.fromkeys(ESI, expected)
    assert _finalize_record(args) == {
        "gate_report_sha256": None,
        "waivers": dict.fromkeys(ESI, WAIVED),
    }


@pytest.mark.parametrize("flags", [None, WAIVED], ids=["unwaived", "waived"])
def test_a_finalize_waiver_record_without_a_gate_report_digest_is_unknown(
    tmp_path, monkeypatch, acs, flags
) -> None:
    # A record saved before records named their gate report holds the waivers
    # alone, by entry id, as 24a417c11d wrote them. Nothing ties it to the
    # report package selects, so the shadow records its waiver as unknown.
    args = _args(tmp_path)
    args.gate_report = tmp_path / "gate_summary.json"
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in ("materialize", "calibrate", "qa"):
        _run_stage(acs, args, stage)
    _run_stage(acs, args, "finalize", report={"run": "only"})
    (acs_directory(args) / "finalize.waivers.json").write_text(
        json.dumps(dict.fromkeys(ESI, flags) if flags else {})
    )

    _run_stage(acs, args, "package")

    expected = [
        "unknown: the finalize stage's waiver record names no gate report, so it "
        f"cannot be matched to the selected gate report {_sha256(args.gate_report)}"
    ]
    assert _comparison(args)["run_completed"] is True
    assert _esi_waived_by(args) == dict.fromkeys(ESI, expected)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        # Truncated.
        ("{", "JSONDecodeError: Expecting property name"),
        # Wrong shape.
        ("[]", "the record is a list, not a mapping"),
        ('{"waivers": [], "gate_report_sha256": null}', "not lists of flags"),
        # Wrong types.
        (
            '{"waivers": {"us_local_esi_premiums_signal": [1]}, '
            '"gate_report_sha256": null}',
            "not lists of flags",
        ),
        ('{"waivers": {}, "gate_report_sha256": 7}', "is not a SHA-256 hex digest"),
        # An entry finalize does not waive, which the battery would accept.
        (
            '{"us_local_take_up": ["--allow-esi-premium-gaps"]}',
            "names entries the finalize stage does not waive: ['us_local_take_up']",
        ),
    ],
    ids=["truncated", "list", "waivers_list", "flag_type", "digest_type", "foreign"],
)
def test_an_unreadable_waiver_record_is_unknown_and_the_comparison_is_rebuilt(
    tmp_path, monkeypatch, acs, capsys, text, error
) -> None:
    # A waiver record the shadow cannot read costs its own entries' waiver,
    # which becomes unknown and names the file and the error. The shadow
    # still rebuilds the comparison from every stage's observations, with
    # its differences.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs, failing={"local_hours_signal"})
    passing = GateResult(name="acs_local_hours_signal", passed=True)
    legacy = {
        entry_id: (lambda: passing)
        for entry_id in ("us_local_hours_finalize", "us_local_hours_package")
    }
    for stage in ("materialize", "calibrate", "qa"):
        _run_stage(acs, args, stage)
    _run_stage(acs, args, "finalize", legacy=legacy)
    (acs_directory(args) / "finalize.waivers.json").write_text(text)

    _run_stage(acs, args, "package", legacy=legacy)

    payload = _comparison(args)
    assert payload["run_completed"] is True
    assert {row["id"] for row in payload["entries"]} == {
        entry.id for entry in us_scoped_gate_manifest("us_local").gates
    }
    differing = [row["id"] for row in payload["differences"]]
    assert differing and all(entry.startswith("us_local_hours") for entry in differing)
    waived_by = {row["id"]: row["waived_by"] for row in payload["entries"]}
    for entry_id in ESI:
        (marker,) = waived_by[entry_id]
        assert marker.startswith(
            "unknown: finalize.waivers.json could not be read as a waiver record ("
        )
        assert error in marker
    assert all(
        flags == [] for entry_id, flags in waived_by.items() if entry_id not in ESI
    )
    assert "finalize.waivers.json could not be read" in capsys.readouterr().out


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


def test_starting_a_stage_clears_the_scope_files_it_outdates(
    tmp_path, monkeypatch, acs
) -> None:
    # Starting a stage discards its own and every later stage's observations,
    # so the scope's report and comparison built from them go at once: a
    # stage killed before its end must not leave a finished run's verdict.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in STAGE_ENTRIES:
        _run_stage(acs, args, stage)
    report, comparison = shadow_file_paths(acs_directory(args), "us_local")
    assert report.is_file()
    assert json.loads(comparison.read_text())["run_completed"] is True
    # The qa stage declares no entry, and it still outdates the later ones.
    with acs._local_gate_shadow_stage(args, "qa"):
        assert not comparison.exists()
        assert not report.exists()
        assert not (acs_directory(args) / "package.observations.json").exists()
    # Its end rebuilds them from the stages left: a run stopped after qa.
    payload = _comparison(args)
    assert payload["run_completed"] is False
    assert payload["stopped_at"] == "after the qa stage"


def test_a_stage_that_saved_nothing_is_never_named_unreached(
    tmp_path, monkeypatch, acs
) -> None:
    # A phase carried in from other processes holds outcomes, not the ids the
    # run evaluated (#1609), so the shadow's own unreached list is the only
    # check here. Finalize's entries ran, but its file is gone, as when its
    # save failed: the package stop must not name them unreached.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs, failing={"stored_inputs"})
    for stage in ("materialize", "calibrate", "qa", "finalize"):
        _run_stage(acs, args, stage)
    (acs_directory(args) / "finalize.observations.json").unlink()
    with pytest.raises(SystemExit, match="hard gate failure"):
        with acs._local_gate_shadow_stage(args, "package"):
            observe_entries("us_local", acs, ("us_local_stored_inputs",))
            raise SystemExit("package: hard gate failure(s): stored inputs")
    payload = _comparison(args)
    assert payload["observations_missing"] == [
        "the finalize stage saved no observations"
    ]
    assert payload["phases_recorded"] == ["assembled"]
    assert payload["blocked_at_phase"] is None


def test_a_difference_is_recorded_without_failing_the_package_stage(
    tmp_path, monkeypatch, acs, capsys
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
    # Today's gates still decide: the package stage finishes.
    _run_stage(acs, args, "package", legacy=legacy)
    payload = _comparison(args)
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is True
    differing = [row["id"] for row in payload["differences"]]
    assert differing and all(entry.startswith("us_local_hours") for entry in differing)
    assert "scope 'us_local' disagrees with today's gates" in capsys.readouterr().err


def test_a_package_stage_finishes_when_its_shadow_files_fail(
    tmp_path, monkeypatch, acs, capsys
) -> None:
    # The shadow never stops a stage: here the package stage's observation
    # files can be neither cleared at its start, saved, nor read back.
    args = _args(tmp_path)
    stand_in_gates(monkeypatch, "us_local", acs)
    for stage in ("materialize", "calibrate", "qa", "finalize"):
        _run_stage(acs, args, stage)

    def unreachable(args, stage):
        raise OSError("observations unreachable")

    monkeypatch.setattr(acs, "_local_stage_observations", unreachable)
    _run_stage(acs, args, "package")
    out = capsys.readouterr().out
    assert "gate battery shadow could not clear earlier files" in out
    assert "gate battery shadow could not be rebuilt" in out
