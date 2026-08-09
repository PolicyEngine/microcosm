import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import microcosm.build.stage_profile as stage_profile
from microcosm.build.stage_profile import profile_stage


def _load_profile(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "stage_profile.json").read_text())


def test__given_successful_stage__then_sampled_peak_and_status_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
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
    assert record["rss_sample_count"] >= 1
    assert math.isfinite(record["wall_seconds"])
    assert record["wall_seconds"] >= 0.0
    assert record["status"] == "succeeded"
    assert record["error"] is None


def test__given_failing_stage__then_failure_is_recorded_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
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
    assert not any(
        thread.name == "microcosm-stage-profile-b" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test__given_separate_stage_invocations__then_profile_is_merged_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
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
    monkeypatch.setattr(
        stage_profile,
        "_read_procfs_current_rss_bytes",
        lambda: None,
    )
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
    assert not any(
        thread.name == "microcosm-stage-profile-c" and thread.is_alive()
        for thread in threading.enumerate()
    )


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
