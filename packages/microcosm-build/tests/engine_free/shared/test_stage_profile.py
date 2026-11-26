import itertools
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import microcosm.build.stage_profile as stage_profile
from microcosm.build.stage_profile import profile_stage


def _load_profile(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "stage_profile.json").read_text())


def _with_readable_procfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the profiler find current RSS in /proc, as on Linux."""

    monkeypatch.setattr(stage_profile, "_read_procfs_current_rss_bytes", lambda: 4096)


def _without_procfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the profiler find no /proc, as on macOS."""

    monkeypatch.setattr(stage_profile, "_read_procfs_current_rss_bytes", lambda: None)


def _with_operating_system(monkeypatch: pytest.MonkeyPatch, sysname: str) -> None:
    monkeypatch.setattr(
        stage_profile.os,
        "uname",
        lambda: SimpleNamespace(sysname=sysname),
    )


def _with_lifetime_peaks(
    monkeypatch: pytest.MonkeyPatch,
    raw_lifetime_peaks: tuple[int, ...],
) -> list[int]:
    """Make getrusage report these ru_maxrss values, one per call, in order.

    Calls after the last value repeat it. Returns the list of values handed out
    so far.
    """

    remaining_peaks = itertools.chain(
        raw_lifetime_peaks, itertools.repeat(raw_lifetime_peaks[-1])
    )
    handed_out: list[int] = []

    def fake_getrusage(who: int) -> SimpleNamespace:
        assert who == stage_profile.resource.RUSAGE_SELF
        raw_peak = next(remaining_peaks)
        handed_out.append(raw_peak)
        return SimpleNamespace(ru_maxrss=raw_peak)

    monkeypatch.setattr(stage_profile.resource, "getrusage", fake_getrusage)
    return handed_out


def _with_ps_reports_kib(
    monkeypatch: pytest.MonkeyPatch,
    rss_kib_per_call: tuple[int, ...],
) -> list[list[str]]:
    """Make ``ps -o rss=`` report these KiB values, one per call, in order.

    Calls after the last value repeat it. Returns the list of ps command lines
    run so far.
    """

    remaining_reports = itertools.chain(
        rss_kib_per_call, itertools.repeat(rss_kib_per_call[-1])
    )
    ps_commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        ps_commands.append(list(command))
        return SimpleNamespace(stdout=f"{next(remaining_reports)}\n")

    monkeypatch.setattr(stage_profile.shutil, "which", lambda _name: "/bin/ps")
    monkeypatch.setattr(stage_profile.subprocess, "run", fake_run)
    return ps_commands


def _profiler_threads_alive(stage_name: str) -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == f"microcosm-stage-profile-{stage_name}" and thread.is_alive()
    ]


def test__given_successful_stage__then_sampled_peak_and_status_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _with_readable_procfs(monkeypatch)
    main_thread_values = iter((100, 200))
    background_sampled = threading.Event()

    def fake_current_rss_bytes() -> int:
        if threading.current_thread() is threading.main_thread():
            return next(main_thread_values)
        background_sampled.set()
        return 500

    monkeypatch.setattr(
        stage_profile,
        "_current_rss_bytes",
        fake_current_rss_bytes,
    )

    # When
    with profile_stage("a", tmp_path, sample_interval_seconds=0.001):
        assert background_sampled.wait(timeout=1.0)

    # Then
    payload = _load_profile(tmp_path)
    record = payload["stages"]["a"]
    assert payload["schema_version"] == 1
    assert record["entry_rss_bytes"] == 100
    assert record["peak_rss_bytes"] == 500
    assert record["exit_rss_bytes"] == 200
    assert record["peak_rss_source"] == "sampled_current_rss"
    assert record["rss_sample_count"] >= 1
    assert math.isfinite(record["wall_seconds"])
    assert record["wall_seconds"] >= 0.0
    assert record["status"] == "succeeded"
    assert record["error"] is None


@pytest.mark.parametrize(
    ("sysname", "raw_lifetime_peaks", "expected_peak_rss_bytes"),
    [
        # Darwin reports ru_maxrss in bytes.
        ("Darwin", (2_000_000, 9_000_000), 9_000_000),
        # Linux reports ru_maxrss in KiB.
        ("Linux", (2_000, 9_000), 9_000 * 1024),
    ],
)
def test__given_no_procfs__then_no_ps_runs_during_stage_and_peak_is_lifetime_peak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sysname: str,
    raw_lifetime_peaks: tuple[int, int],
    expected_peak_rss_bytes: int,
) -> None:
    # Given
    _without_procfs(monkeypatch)
    _with_operating_system(monkeypatch, sysname)
    lifetime_peaks_read = _with_lifetime_peaks(monkeypatch, raw_lifetime_peaks)
    # One ps report for the entry RSS and one for the exit RSS, nothing more.
    ps_commands = _with_ps_reports_kib(monkeypatch, (1_000, 1_200))

    # When
    with profile_stage("macos", tmp_path, sample_interval_seconds=0.001):
        ps_commands_at_stage_start = len(ps_commands)
        threads_during_stage = _profiler_threads_alive("macos")
        # Long enough for a 0.001 s sampler to run ps many times.
        time.sleep(0.05)
        ps_commands_at_stage_end = len(ps_commands)

    # Then
    record = _load_profile(tmp_path)["stages"]["macos"]
    assert threads_during_stage == []
    assert ps_commands_at_stage_start == ps_commands_at_stage_end == 1
    assert len(ps_commands) == 2
    assert lifetime_peaks_read == list(raw_lifetime_peaks)
    assert record["entry_rss_bytes"] == 1_000 * 1024
    assert record["exit_rss_bytes"] == 1_200 * 1024
    assert record["peak_rss_bytes"] == expected_peak_rss_bytes
    assert record["peak_rss_source"] == "process_peak_rss_reached_during_stage"
    assert record["rss_sample_count"] == 0
    assert record["sampling_error"] is None
    assert record["status"] == "succeeded"


def test__given_no_procfs_and_lifetime_peak_set_before_stage__then_record_says_so(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: the process reached its lifetime peak before the stage began, for
    # example during imports or in an earlier profiled stage.
    _without_procfs(monkeypatch)
    _with_operating_system(monkeypatch, "Darwin")
    _with_lifetime_peaks(monkeypatch, (9_000_000, 9_000_000))
    _with_ps_reports_kib(monkeypatch, (1_000, 1_200))

    # When
    with profile_stage("after_imports", tmp_path, sample_interval_seconds=0.001):
        pass

    # Then
    record = _load_profile(tmp_path)["stages"]["after_imports"]
    assert record["peak_rss_bytes"] == 9_000_000
    assert record["peak_rss_source"] == "process_peak_rss_reached_before_stage"
    assert record["sampling_error"] is None


def test__given_no_procfs_and_lifetime_peak_below_exit_rss__then_peak_is_exit_rss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: readers require peak >= entry and peak >= exit, even when ps and
    # getrusage disagree.
    _without_procfs(monkeypatch)
    _with_operating_system(monkeypatch, "Darwin")
    _with_lifetime_peaks(monkeypatch, (500_000, 600_000))
    _with_ps_reports_kib(monkeypatch, (1_000, 1_200))

    # When
    with profile_stage("disagree", tmp_path, sample_interval_seconds=0.001):
        pass

    # Then
    record = _load_profile(tmp_path)["stages"]["disagree"]
    assert record["peak_rss_bytes"] == 1_200 * 1024
    assert record["peak_rss_bytes"] >= record["entry_rss_bytes"]
    assert record["peak_rss_bytes"] >= record["exit_rss_bytes"]


def test__given_no_procfs_and_unreadable_lifetime_peak__then_error_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _without_procfs(monkeypatch)
    _with_operating_system(monkeypatch, "Darwin")
    _with_lifetime_peaks(monkeypatch, (0, 0))
    _with_ps_reports_kib(monkeypatch, (1_000, 1_200))

    # When
    with profile_stage("no_rusage", tmp_path, sample_interval_seconds=0.001):
        pass

    # Then
    record = _load_profile(tmp_path)["stages"]["no_rusage"]
    assert record["peak_rss_bytes"] == 1_200 * 1024
    assert record["peak_rss_source"] == "entry_and_exit_rss"
    assert record["sampling_error"] == {
        "message": "resource reported non-positive peak RSS: 0",
        "type": "builtins.RuntimeError",
    }
    assert record["status"] == "succeeded"


def test__given_no_procfs_and_failing_stage__then_failure_is_recorded_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _without_procfs(monkeypatch)
    _with_operating_system(monkeypatch, "Darwin")
    _with_lifetime_peaks(monkeypatch, (2_000_000, 3_000_000))
    _with_ps_reports_kib(monkeypatch, (1_000, 1_200))

    # When
    with pytest.raises(ValueError, match="stage exploded"):
        with profile_stage("macos_failure", tmp_path, sample_interval_seconds=0.001):
            raise ValueError("stage exploded")

    # Then
    record = _load_profile(tmp_path)["stages"]["macos_failure"]
    assert record["status"] == "failed"
    assert record["error"] == {
        "message": "stage exploded",
        "type": "builtins.ValueError",
    }
    assert record["peak_rss_bytes"] == 3_000_000
    assert record["peak_rss_source"] == "process_peak_rss_reached_during_stage"


def test__given_version_1_profile_from_earlier_run__then_new_record_is_merged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a resumed run's checkpoint directory already holds a profile
    # written before records carried peak_rss_source.
    earlier_record = {
        "entry_rss_bytes": 100,
        "error": None,
        "exit_rss_bytes": 200,
        "peak_rss_bytes": 500,
        "rss_sample_count": 3,
        "sampling_error": None,
        "stage": "source_construction",
        "status": "succeeded",
        "wall_seconds": 1.5,
    }
    (tmp_path / "stage_profile.json").write_text(
        json.dumps(
            {"schema_version": 1, "stages": {"source_construction": earlier_record}}
        )
    )
    _without_procfs(monkeypatch)
    _with_operating_system(monkeypatch, "Darwin")
    _with_lifetime_peaks(monkeypatch, (2_000_000, 3_000_000))
    _with_ps_reports_kib(monkeypatch, (1_000, 1_200))

    # When
    with profile_stage("pre_clone_enrichment", tmp_path, sample_interval_seconds=0.001):
        pass

    # Then
    payload = _load_profile(tmp_path)
    assert payload["schema_version"] == 1
    assert payload["stages"]["source_construction"] == earlier_record
    new_record = payload["stages"]["pre_clone_enrichment"]
    assert new_record["peak_rss_bytes"] == 3_000_000
    assert new_record["peak_rss_source"] == "process_peak_rss_reached_during_stage"


def test__given_failing_stage__then_failure_is_recorded_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _with_readable_procfs(monkeypatch)
    monkeypatch.setattr(stage_profile, "_current_rss_bytes", lambda: 300)

    # When
    with pytest.raises(ValueError, match="stage exploded"):
        with profile_stage("b", tmp_path, sample_interval_seconds=0.001):
            raise ValueError("stage exploded")

    # Then
    record = _load_profile(tmp_path)["stages"]["b"]
    assert record["status"] == "failed"
    assert record["error"] == {
        "message": "stage exploded",
        "type": "builtins.ValueError",
    }
    assert _profiler_threads_alive("b") == []


def test__given_separate_stage_invocations__then_profile_is_merged_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _with_readable_procfs(monkeypatch)
    monkeypatch.setattr(stage_profile, "_current_rss_bytes", lambda: 400)

    # When
    with profile_stage("b", tmp_path, sample_interval_seconds=0.001):
        pass
    with profile_stage("a", tmp_path, sample_interval_seconds=0.001):
        pass

    # Then
    profile_path = tmp_path / "stage_profile.json"
    serialized = profile_path.read_text()
    payload = json.loads(serialized)
    assert list(payload["stages"]) == ["a", "b"]
    assert serialized.index('"a"') < serialized.index('"b"')
    assert serialized.endswith("\n")
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert all(
        math.isfinite(record["wall_seconds"]) for record in payload["stages"].values()
    )


def test__given_atomic_replace_failure__then_previous_profile_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _with_readable_procfs(monkeypatch)
    monkeypatch.setattr(stage_profile, "_current_rss_bytes", lambda: 500)
    with profile_stage("a", tmp_path, sample_interval_seconds=0.001):
        pass
    profile_path = tmp_path / "stage_profile.json"
    previous_contents = profile_path.read_bytes()

    def fail_replace(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(stage_profile.os, "replace", fail_replace)

    # When
    with pytest.raises(OSError, match="replace failed"):
        with profile_stage("b", tmp_path, sample_interval_seconds=0.001):
            pass

    # Then
    assert profile_path.read_bytes() == previous_contents
    assert list(tmp_path.glob(".stage_profile.json.*.tmp")) == []


def test__given_unavailable_procfs__then_ps_current_rss_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    _without_procfs(monkeypatch)
    monkeypatch.setattr(stage_profile.shutil, "which", lambda _name: "/bin/ps")
    monkeypatch.setattr(
        stage_profile.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="123\n"),
    )

    # When
    current_rss_bytes = stage_profile.current_rss_bytes()

    # Then
    assert current_rss_bytes == 123 * 1024


def test__given_exit_rss_failure__then_sampler_thread_is_still_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    _with_readable_procfs(monkeypatch)
    main_thread_calls = 0

    def fail_on_exit_rss() -> int:
        nonlocal main_thread_calls
        if threading.current_thread() is not threading.main_thread():
            return 600
        main_thread_calls += 1
        if main_thread_calls == 1:
            return 600
        raise RuntimeError("exit RSS unavailable")

    monkeypatch.setattr(stage_profile, "_current_rss_bytes", fail_on_exit_rss)

    # When
    with pytest.raises(RuntimeError, match="exit RSS unavailable"):
        with profile_stage("c", tmp_path, sample_interval_seconds=0.001):
            pass

    # Then
    assert _profiler_threads_alive("c") == []


@pytest.mark.parametrize("sample_interval_seconds", [0.0, -1.0, math.nan])
def test__given_invalid_sample_interval__then_value_error_is_raised(
    sample_interval_seconds: float,
    tmp_path: Path,
) -> None:
    # Given / When / Then
    with pytest.raises(ValueError, match="sample_interval_seconds"):
        with profile_stage(
            "a",
            tmp_path,
            sample_interval_seconds=sample_interval_seconds,
        ):
            pass
