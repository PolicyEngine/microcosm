"""Test Microcosm's adapter with provider APIs replaced, not hosted services."""

from types import SimpleNamespace

import pytest

from microcosm.build import telemetry_emitter as adapter

START = adapter.LocalTelemetryEmitter.start.__func__


def test_selects_both_workers_once_with_shared_path_and_build_identity(
    monkeypatch, tmp_path
):
    calls, messages, closed = [], [], []
    transport = SimpleNamespace(send=messages.append)

    def for_module(name):
        assert name == "telemetry"
        return transport

    handle = SimpleNamespace(
        client=SimpleNamespace(for_module=for_module), close=lambda: closed.append(True)
    )

    def start(modules, **options):
        calls.append((modules, options))
        return handle

    monkeypatch.setattr(adapter, "start_services", start)
    monkeypatch.setattr(adapter, "runtime_identity", lambda: {"commit": "synthetic"})
    path = tmp_path / "jobs.sqlite3"
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=path,
        heartbeat_seconds=123,
    )
    assert len(calls) == 1
    modules, options = calls[0]
    assert set(modules) == {"telemetry", "orrery"}
    assert all(m["spool_path"] == str(path) for m in modules.values())
    assert "registration" not in modules["orrery"]
    assert modules["telemetry"]["registration"]["run_id"] == "run-one"
    assert modules["telemetry"]["heartbeat_seconds"] == 123
    assert options["startup_timeout"] == 3
    assert messages[0]["details"] == {"identity": {"commit": "synthetic"}}
    emitter.complete()
    emitter.close()
    assert closed == [True]
    assert messages[-1]["status"] == "completed"


def test_disabled_telemetry_still_delegates_durable_graph_enqueue(
    monkeypatch, tmp_path
):
    calls = []

    def publish(directory, inventory, **options):
        calls.append((directory, inventory, options))
        assert options["available"]() is False
        return {"status": "pending", "publication_id": "independent"}

    monkeypatch.setattr(adapter, "publish_graph", publish)
    path = tmp_path / "jobs.sqlite3"
    emitter = adapter.LocalTelemetryEmitter(
        run=adapter.TelemetryRun("run-one", "uk", "full"),
        transport=None,
        spool_path=path,
    )
    value = {"publication_id": "independent"}
    receipt = emitter.publish_graph(tmp_path, value, wait_seconds=0)
    assert receipt["status"] == "pending"
    assert calls[0][0:2] == (tmp_path, value)
    assert calls[0][2]["spool_path"] == path
    assert calls[0][2]["wait_seconds"] == 0


def test_start_failure_is_nonfatal_and_does_not_log_exception_payload(
    monkeypatch, capsys, tmp_path
):
    def fail(*args, **kwargs):
        raise RuntimeError("private credential should not be logged")

    monkeypatch.setattr(adapter, "start_services", fail)
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="run-one",
        country_code="uk",
        pipeline="full",
        spool_path=tmp_path / "jobs.sqlite3",
    )
    assert not emitter.available
    warning = capsys.readouterr().err
    assert "RuntimeError" in warning
    assert "private credential" not in warning


@pytest.mark.parametrize("operation", ["complete", "close"])
def test_default_test_emitter_uses_provider_lifecycle_in_memory(
    fake_telemetry_emitters, operation
):
    emitter = adapter.LocalTelemetryEmitter.start(
        run_id="run-one", country_code="uk", pipeline="full"
    )
    emitter.stage("one")
    getattr(emitter, operation)()
    assert not emitter.available
    assert emitter.events[0]["stage_id"] == "one"


def test_fixture_intercepts_preimported_build_entrypoint(
    monkeypatch, fake_telemetry_emitters
):
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    def fail(*args, **kwargs):
        pytest.fail("test started a real provider")

    monkeypatch.setattr(adapter, "start_services", fail)
    for start in (
        rowwise_staging.start_local_telemetry_emitter_service,
        adapter.LocalTelemetryEmitter.start,
    ):
        emitter = start(run_id="fake", country_code="GB", pipeline="test")
        assert isinstance(emitter, FakeTelemetryEmitter)
        assert emitter in fake_telemetry_emitters


def test_fixture_isolates_cached_huggingface_credentials(tmp_path):
    from pathlib import Path

    from huggingface_hub import constants, get_token

    assert Path(constants.HF_TOKEN_PATH).is_relative_to(tmp_path)
    assert get_token() is None


@pytest.mark.parametrize("failure", ["default_spool_path", "runtime_identity"])
def test_metadata_or_cache_failure_never_starts_a_service(monkeypatch, failure):
    def fail():
        raise OSError("private setup detail")

    monkeypatch.setattr(adapter, failure, fail)
    monkeypatch.setattr(
        adapter, "start_services", lambda *a, **kw: pytest.fail("started after failure")
    )
    emitter = START(
        adapter.LocalTelemetryEmitter,
        run_id="failed",
        country_code="uk",
        pipeline="full",
    )
    assert not emitter.available
