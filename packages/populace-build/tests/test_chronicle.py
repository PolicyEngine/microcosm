"""Contracts for the direct #628-compatible Chronicle spool."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

import pytest

import populace.build.chronicle as chronicle
from populace.build.chronicle import (
    CHRONICLE_ROW_FIELDS,
    ChronicleRow,
    canonical_json_bytes,
    load_chronicle_row,
    reconcile_spool,
    record_build_attempt,
)

ROOT = Path(__file__).resolve().parents[3]
MIGRATION = ROOT / "supabase/migrations/20260805000000_chronicle.sql"


@pytest.fixture(autouse=True)
def _spool_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never inherit operator Chronicle credentials."""

    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)


def _row_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "build_id": "fixture-build-1",
        "ts": "2026-08-04T15:06:00-04:00",
        "pipeline": "fixture-pipeline",
        "rung": "f010",
        "seed": 628,
        "code_pin": "1c1fc717",
        "input_pins_digest": "1" * 64,
        "identity_digest": "2" * 64,
        "phases_reached": ["assembled", "simulated"],
        "gate_verdicts": {
            "agreement": {
                "verdict": "failed",
                "receipt": "receipt://fixture/agreement.json",
                "diagnostics": {"ratio": 1.0, "unicode": "café"},
            }
        },
        "wall_seconds": 12.5,
        "cost_usd": 1.0,
        "artifact_location": None,
        "disposition": "failed",
        "prediction_id": "p001",
        "prev_row_digest": None,
    }
    values.update(overrides)
    return values


def test_row_schema_json_round_trip_matches_628_golden() -> None:
    row = ChronicleRow.create(**_row_kwargs())

    restored = ChronicleRow.from_mapping(json.loads(row.to_json_line()))

    assert restored == row
    assert frozenset(row.to_mapping()) == CHRONICLE_ROW_FIELDS
    assert restored.ts == "2026-08-04T19:06:00.000000Z"
    assert restored.row_digest == (
        "80a01b5cdefeeed6a8acd36dfa06b1e4506f4853c2786101d3c3ba414cd8a927"
    )


def test_sql_schema_round_trip_matches_python_hash_surface() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    builds = sql.split("CREATE TABLE chronicle.builds (", 1)[1].split("\n);", 1)[0]
    for field in CHRONICLE_ROW_FIELDS:
        assert re.search(rf"^    {field}\s", builds, flags=re.MULTILINE), field

    payload = sql.split("CREATE OR REPLACE FUNCTION chronicle.build_hash_payload", 1)[
        1
    ].split("$function$;", 1)[0]
    payload_fields = set(re.findall(r"'([a-z_]+)',\s+p_build\.", payload))
    payload_fields.add("ts")
    assert payload_fields == CHRONICLE_ROW_FIELDS - {
        "prev_row_digest",
        "row_digest",
    }
    assert "trim_scale((p_value #>> '{}')::numeric)::text" in sql
    assert "rung IN ('f001', 'f010', 'f100')" in sql
    assert "CHECK (chronicle.valid_build_phases(phases_reached))" in builds
    assert "CHECK (chronicle.valid_gate_verdicts(gate_verdicts))" in builds
    assert "phases_reached jsonb NOT NULL DEFAULT" not in builds
    assert "gate_verdicts jsonb NOT NULL DEFAULT" not in builds
    function_grants = sql.split("GRANT EXECUTE ON FUNCTION", 1)[1].split(
        "GRANT INSERT ON", 1
    )[0]
    assert "chronicle.nonempty_trimmed_text(text)" in function_grants
    assert "chronicle.valid_build_phases(jsonb)" in function_grants
    assert "chronicle.valid_gate_verdicts(jsonb)" in function_grants
    assert "TO chronicle_writer, chronicle_break_glass_admin" in function_grants

    text_helper = sql.split(
        "CREATE OR REPLACE FUNCTION chronicle.nonempty_trimmed_text",
        1,
    )[1].split("$function$;", 1)[0]
    sql_whitespace = {
        int(codepoint, 16) for codepoint in re.findall(r"\\([0-9A-F]{4})", text_helper)
    }
    python_whitespace = {
        codepoint for codepoint in range(0x110000) if chr(codepoint).isspace()
    }
    assert sql_whitespace == python_whitespace

    public_view = sql.split("CREATE VIEW chronicle.builds_public", 1)[1].split(
        "FROM chronicle.builds;", 1
    )[0]
    assert "cost_usd" not in public_view
    assert "gate_verdicts" not in public_view
    assert "WHEN disposition IN ('published', 'certified')" in public_view
    assert "ELSE NULL" in public_view
    assert "GRANT INSERT ON chronicle.builds, chronicle.predictions" in sql
    assert "TO chronicle_writer" in sql
    assert "GRANT SELECT ON chronicle.builds" in sql
    assert "TO chronicle_exporter" in sql
    assert "CREATE POLICY builds_exporter_select" in sql
    assert "CREATE POLICY predictions_exporter_select" not in sql
    assert "GRANT chronicle_writer, chronicle_exporter TO authenticator" in sql


def test_canonical_json_matches_sql_number_and_unicode_vector() -> None:
    value = {"z": 1.0, "a": ["é", 1e-3, None], "n": -0.0}

    assert canonical_json_bytes(value).decode() == (
        '{"a":["é",0.001,null],"n":0,"z":1}'
    )


@pytest.mark.parametrize(
    "value",
    [
        {"nested": "diagnostic\x00detail"},
        {"diagnostic\x00key": "detail"},
    ],
)
def test_canonical_json_rejects_postgresql_incompatible_nul(
    value: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="NUL"):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("build_id", "bad build id", "build_id"),
        ("ts", "2026-08-04T19:06:00", "UTC offset"),
        ("pipeline", " fixture ", "pipeline"),
        ("pipeline", "fixture\x00pipeline", "NUL"),
        ("rung", "1/10", "#624 fraction token"),
        ("rung", "0.1", "#624 fraction token"),
        ("rung", "f020", "#624 fraction token"),
        ("seed", True, "seed"),
        ("seed", -1, "seed"),
        ("code_pin", "", "code_pin"),
        ("input_pins_digest", "not-a-digest", "input_pins_digest"),
        ("identity_digest", "A" * 64, "identity_digest"),
        ("phases_reached", [], "phases_reached"),
        ("phases_reached", "assembled", "phases_reached"),
        ("phases_reached", ["assembled", "assembled"], "duplicate"),
        ("gate_verdicts", {}, "gate_verdicts"),
        ("gate_verdicts", {"agreement": "failed"}, "must be an object"),
        ("gate_verdicts", {"agreement": {"verdict": "failed"}}, "receipt"),
        ("wall_seconds", -0.1, "wall_seconds"),
        ("cost_usd", float("nan"), "cost_usd"),
        ("disposition", "succeeded", "disposition"),
        ("prediction_id", " ", "prediction_id"),
        ("prev_row_digest", "not-a-digest", "prev_row_digest"),
    ],
)
def test_row_validation_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ChronicleRow.create(**_row_kwargs(**{field: value}))


def test_schema_rejects_missing_and_extra_fields_by_name() -> None:
    mapping = ChronicleRow.create(**_row_kwargs()).to_mapping()
    del mapping["identity_digest"]
    mapping["fraction_token"] = "f010"

    with pytest.raises(
        ValueError,
        match=r"missing=\['identity_digest'\].*extra=\['fraction_token'\]",
    ):
        ChronicleRow.from_mapping(mapping)


def test_row_rejects_nul_in_nested_gate_diagnostics() -> None:
    gate_verdicts = {
        "agreement": {
            "verdict": "failed",
            "receipt": "receipt://fixture/agreement.json",
            "diagnostics": {"detail": "unreconcilable\x00postgresql-text"},
        }
    }

    with pytest.raises(ValueError, match="NUL"):
        ChronicleRow.create(**_row_kwargs(gate_verdicts=gate_verdicts))


def test_wrong_supplied_row_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="row_digest"):
        ChronicleRow.create(**_row_kwargs(), row_digest="f" * 64)


def test_published_row_requires_an_artifact_location() -> None:
    with pytest.raises(ValueError, match="artifact_location is required"):
        ChronicleRow.create(**_row_kwargs(disposition="published"))


@pytest.mark.parametrize("rung", ["f001", "f010", "f100"])
def test_standard_scale_rungs_are_accepted(rung: str) -> None:
    assert ChronicleRow.create(**_row_kwargs(rung=rung)).rung == rung


def test_predecessor_is_bound_once_into_the_row_digest() -> None:
    genesis = ChronicleRow.create(**_row_kwargs())
    successor = ChronicleRow.create(
        **_row_kwargs(
            build_id="fixture-build-2",
            prev_row_digest=genesis.row_digest,
        )
    )

    assert successor.prev_row_digest == genesis.row_digest
    assert successor.row_digest != genesis.row_digest


def test_spool_round_trip_authenticates_contents_and_filename(tmp_path: Path) -> None:
    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert result.spool_path == tmp_path / f"{result.row.row_digest}.json"
    assert load_chronicle_row(result.spool_path) == result.row

    tampered = json.loads(result.spool_path.read_text(encoding="utf-8"))
    tampered["pipeline"] = "tampered"
    result.spool_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match=r"row_digest.*does not match"):
        load_chronicle_row(result.spool_path)


def test_spool_filename_mutation_is_rejected(tmp_path: Path) -> None:
    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    wrong_name = tmp_path / f"{'f' * 64}.json"
    wrong_name.write_bytes(result.spool_path.read_bytes())

    with pytest.raises(ValueError, match="filename does not match row_digest"):
        load_chronicle_row(wrong_name)


def test_spool_write_is_atomic_and_durable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(chronicle.os, "fsync", tracked_fsync)
    monkeypatch.setattr(chronicle.os, "replace", tracked_replace)

    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path / "spool")

    assert result.spool_path.exists()
    assert list((tmp_path / "spool").glob(".*.tmp")) == []
    assert events == [
        ("fsync", "file"),
        ("replace", result.spool_path.name),
        ("fsync", "directory"),
    ]


def test_spool_retry_completes_parent_fsync_after_interrupted_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_fsync_parent = chronicle._fsync_parent_directory
    parent_fsync_calls = 0

    def fail_first_parent_fsync(path: Path) -> None:
        nonlocal parent_fsync_calls
        parent_fsync_calls += 1
        if parent_fsync_calls == 1:
            raise OSError("injected directory fsync failure")
        real_fsync_parent(path)

    monkeypatch.setattr(
        chronicle,
        "_fsync_parent_directory",
        fail_first_parent_fsync,
    )

    with pytest.raises(OSError, match="injected directory fsync failure"):
        record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    persisted = next(tmp_path.glob("*.json"))

    retry = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert retry.spool_path == persisted
    assert parent_fsync_calls == 2


def test_identical_retry_is_idempotent(tmp_path: Path) -> None:
    first = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    original_bytes = first.spool_path.read_bytes()

    second = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert second == first
    assert second.spool_path.read_bytes() == original_bytes
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_nested_tuple_gate_data_normalizes_for_exact_retry(tmp_path: Path) -> None:
    gate_verdicts = {
        "agreement": {
            "verdict": "failed",
            "receipt": "receipt://fixture/agreement.json",
            "bounds": (1, 2),
            "nested": ({"values": (3, 4)},),
        }
    }

    first = record_build_attempt(
        **_row_kwargs(gate_verdicts=gate_verdicts),
        spool_dir=tmp_path,
    )
    second = record_build_attempt(
        **_row_kwargs(gate_verdicts=gate_verdicts),
        spool_dir=tmp_path,
    )

    assert second == first
    assert second.row.gate_verdicts["agreement"]["bounds"] == [1, 2]
    assert second.row.gate_verdicts["agreement"]["nested"] == [{"values": [3, 4]}]


def test_missing_remote_environment_is_spool_only(tmp_path: Path) -> None:
    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert result.spool_path.exists()
    assert result.posted is False
    assert result.remote_error is None


class _Response:
    status = 201

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_remote_insert_happens_after_spool_and_uses_chronicle_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        requests.append(request)
        assert timeout == 3.0
        assert len(list(tmp_path.glob("*.json"))) == 1
        return _Response()

    monkeypatch.setattr(chronicle, "urlopen", fake_urlopen)

    result = record_build_attempt(
        **_row_kwargs(),
        spool_dir=tmp_path,
        timeout=3.0,
    )

    assert result.posted is True
    assert result.remote_error is None
    request = requests[0]
    assert request.full_url == (
        "https://fixture.supabase.co/rest/v1/builds?on_conflict=build_id"
    )
    assert request.method == "POST"
    assert request.headers["Apikey"] == "project-api-key"
    assert request.headers["Authorization"] == "Bearer writer-jwt"
    assert request.headers["Content-profile"] == "chronicle"
    assert request.headers["Prefer"] == ("resolution=ignore-duplicates,return=minimal")
    assert json.loads(request.data) == result.row.to_mapping()


def test_remote_failure_never_blocks_a_build_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")

    def fail(*_args: object, **_kwargs: object) -> _Response:
        raise OSError("offline")

    monkeypatch.setattr(chronicle, "urlopen", fail)

    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert result.posted is False
    assert result.remote_error == "Supabase insert failed: offline"
    assert load_chronicle_row(result.spool_path) == result.row


def test_insecure_remote_url_never_blocks_a_build_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POPULACE_LEDGER_URL", "http://not-loopback.example")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")

    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert result.posted is False
    assert "must use HTTPS" in (result.remote_error or "")
    assert load_chronicle_row(result.spool_path) == result.row


def test_remote_redirects_are_never_followed() -> None:
    handler = chronicle._NoRedirectHandler()
    request = chronicle.Request(
        "https://fixture.supabase.co/rest/v1/builds",
        headers={"Authorization": "Bearer writer-jwt"},
    )

    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "http://attacker.example/collect",
    )

    assert redirected is None


def test_reconcile_posts_in_chain_order_and_removes_acknowledged_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    second = record_build_attempt(
        **_row_kwargs(
            build_id="fixture-build-2",
            prev_row_digest=first.row.row_digest,
        ),
        spool_dir=tmp_path,
    )
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co/rest/v1")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    posted_ids: list[str] = []

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        del timeout
        posted_ids.append(json.loads(request.data)["build_id"])
        return _Response()

    monkeypatch.setattr(chronicle, "urlopen", fake_urlopen)

    receipt = reconcile_spool(tmp_path)

    assert posted_ids == [first.row.build_id, second.row.build_id]
    assert receipt == chronicle.ReconcileResult(2, 2, 0, ())
    assert list(tmp_path.glob("*.json")) == []


def test_reconcile_stops_at_first_failure_and_retains_chain_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    second = record_build_attempt(
        **_row_kwargs(
            build_id="fixture-build-2",
            prev_row_digest=first.row.row_digest,
        ),
        spool_dir=tmp_path,
    )
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    attempts = 0

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        nonlocal attempts
        del request, timeout
        attempts += 1
        if attempts == 2:
            raise OSError("offline")
        return _Response()

    monkeypatch.setattr(chronicle, "urlopen", fake_urlopen)

    receipt = reconcile_spool(tmp_path)

    assert receipt.attempted == 2
    assert receipt.posted == 1
    assert receipt.retained == 1
    assert "fixture-build-2" in receipt.errors[0]
    assert not first.spool_path.exists()
    assert second.spool_path.exists()


def test_reconcile_reports_remote_ack_when_cleanup_fsync_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    monkeypatch.setattr(chronicle, "urlopen", lambda *_args, **_kwargs: _Response())

    def fail_parent_fsync(_path: Path) -> None:
        raise OSError("injected cleanup fsync failure")

    monkeypatch.setattr(chronicle, "_fsync_parent_directory", fail_parent_fsync)

    receipt = reconcile_spool(tmp_path)

    assert receipt.attempted == 1
    assert receipt.posted == 1
    assert receipt.retained == 0
    assert "cleanup fsync failure" in receipt.errors[0]
    assert not result.spool_path.exists()
