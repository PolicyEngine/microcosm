"""CLI contracts for the append-only Logbook archive."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse

import pytest

from microcosm.build.logbook import LogbookRow, load_logbook_file

ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = ROOT / "tools/logbook.py"


@pytest.fixture(autouse=True)
def _no_inherited_logbook_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_EXPORT_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)


def _row(
    build_id: str,
    *,
    predecessor: str | None,
    minute: int,
    rung: str = "f010",
    disposition: str = "failed",
) -> LogbookRow:
    artifact = (
        f"hf://datasets/policyengine/populace-us@{build_id}"
        if disposition in {"published", "certified"}
        else None
    )
    return LogbookRow.create(
        build_id=build_id,
        ts=f"2026-08-05T12:{minute:02d}:00Z",
        pipeline="fixture-pipeline",
        rung=rung,
        seed=628,
        code_pin="1c1fc717",
        input_pins_digest="1" * 64,
        identity_digest="2" * 64,
        phases_reached=["assembled", "simulated"],
        gate_verdicts={
            "agreement": {
                "verdict": "failed",
                "receipt": "receipt://fixture/agreement.json",
            }
        },
        wall_seconds=12.5,
        cost_usd=1.0,
        artifact_location=artifact,
        disposition=disposition,
        prediction_id=None,
        prev_row_digest=predecessor,
    )


def _chain() -> tuple[LogbookRow, LogbookRow, LogbookRow]:
    first = _row("fixture-build-1", predecessor=None, minute=1)
    second = _row(
        "fixture-build-2",
        predecessor=first.row_digest,
        minute=2,
        rung="f100",
        disposition="published",
    )
    third = _row(
        "fixture-build-3",
        predecessor=second.row_digest,
        minute=3,
        rung="f100",
        disposition="certified",
    )
    return first, second, third


def _write_jsonl(path: Path, rows: tuple[LogbookRow, ...]) -> None:
    path.write_text("".join(row.to_json_line() for row in rows), encoding="utf-8")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("logbook_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_the_ratified_root_paths() -> None:
    cli = _load_cli()

    assert cli.DEFAULT_ARCHIVE == ROOT / "logbook.jsonl"
    assert cli.DEFAULT_SOURCE == ROOT / "logbook-spool"


def test_cli_validate_and_filtered_render(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "logbook.jsonl"
    _write_jsonl(archive, _chain())
    cli = _load_cli()

    assert cli.main(["validate", "--archive", str(archive)]) == 0
    assert "validated 3 Logbook rows" in capsys.readouterr().out

    assert (
        cli.main(
            [
                "render",
                "--archive",
                str(archive),
                "--rung",
                "f100",
                "--disposition",
                "certified",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "fixture-build-3" in output
    assert "fixture-build-1" not in output
    assert "fixture-build-2" not in output
    assert "cost_usd" not in output


def test_cli_export_appends_jsonl_source_suffix_idempotently(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first, second, third = _chain()
    archive = tmp_path / "logbook.jsonl"
    source = tmp_path / "source.jsonl"
    _write_jsonl(archive, (first,))
    _write_jsonl(source, (first, second, third))
    cli = _load_cli()

    command = [
        "export",
        "--archive",
        str(archive),
        "--source",
        str(source),
    ]
    assert cli.main(command) == 0
    assert "exported 2 new Logbook rows" in capsys.readouterr().out
    assert load_logbook_file(archive) == (first, second, third)

    assert cli.main(command) == 0
    assert "exported 0 new Logbook rows" in capsys.readouterr().out
    assert load_logbook_file(archive) == (first, second, third)


def test_cli_export_chain_orders_a_spool_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _chain()
    spool = tmp_path / "spool"
    spool.mkdir()
    for row in reversed(rows):
        (spool / f"{row.row_digest}.json").write_text(
            row.to_json_line(),
            encoding="utf-8",
        )
    archive = tmp_path / "logbook.jsonl"
    cli = _load_cli()

    assert (
        cli.main(
            [
                "export",
                "--archive",
                str(archive),
                "--source",
                str(spool),
            ]
        )
        == 0
    )

    assert "exported 3 new Logbook rows" in capsys.readouterr().out
    assert load_logbook_file(archive) == rows


def test_cli_export_divergence_fails_closed_without_modifying_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first, _, _ = _chain()
    archive = tmp_path / "logbook.jsonl"
    _write_jsonl(archive, (first,))
    before = archive.read_bytes()
    divergent = _row(
        "divergent-build",
        predecessor="f" * 64,
        minute=4,
    )
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, (divergent,))
    cli = _load_cli()

    assert (
        cli.main(
            [
                "export",
                "--archive",
                str(archive),
                "--source",
                str(source),
            ]
        )
        == 1
    )

    assert "logbook export failed" in capsys.readouterr().err
    assert archive.read_bytes() == before


class _RemoteResponse:
    status = 206

    def __init__(
        self,
        rows: tuple[LogbookRow, ...],
        *,
        start: int = 0,
        total: int | None = None,
    ) -> None:
        self._payload = json.dumps([row.to_mapping() for row in rows]).encode()
        total = len(rows) if total is None else total
        self.headers = {"Content-Range": f"{start}-{start + len(rows) - 1}/{total}"}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _RemoteResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_cli_remote_export_uses_distinct_read_only_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _chain()
    archive = tmp_path / "logbook.jsonl"
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt-must-not-be-used")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: float) -> _RemoteResponse:
        requests.append(request)
        assert timeout == 30.0
        return _RemoteResponse(rows)

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    assert cli.main(["export", "--remote", "--archive", str(archive)]) == 0

    assert "exported 3 new Logbook rows" in capsys.readouterr().out
    assert load_logbook_file(archive) == rows
    request = requests[0]
    assert request.headers["Accept-profile"] == "logbook"
    assert request.headers["Apikey"] == "project-api-key"
    assert request.headers["Authorization"] == "Bearer exporter-jwt"
    query = parse_qs(urlparse(request.full_url).query)
    assert query["order"] == ["ts.asc,build_id.asc"]
    assert query["limit"] == [str(cli.REMOTE_PAGE_SIZE)]
    assert query["offset"] == ["0"]


def test_cli_remote_export_refuses_the_writer_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")

    result = cli.main(
        ["export", "--remote", "--archive", str(tmp_path / "logbook.jsonl")]
    )

    assert result == 1
    assert "POPULACE_LEDGER_EXPORT_KEY" in capsys.readouterr().err


def test_cli_remote_export_paginates_before_chain_ordering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _chain()
    archive = tmp_path / "logbook.jsonl"
    cli = _load_cli()
    monkeypatch.setattr(cli, "REMOTE_PAGE_SIZE", 2)
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")
    offsets: list[int] = []

    def fake_urlopen(request: object, *, timeout: float) -> _RemoteResponse:
        del timeout
        query = parse_qs(urlparse(request.full_url).query)
        offset = int(query["offset"][0])
        limit = int(query["limit"][0])
        offsets.append(offset)
        return _RemoteResponse(
            rows[offset : offset + limit],
            start=offset,
            total=len(rows),
        )

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    assert cli.main(["export", "--remote", "--archive", str(archive)]) == 0

    assert offsets == [0, 2]
    assert load_logbook_file(archive) == rows


def test_cli_remote_export_rejects_insecure_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "http://not-loopback.example")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")

    def unexpected_request(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("insecure URL must fail before a request")

    monkeypatch.setattr(cli, "urlopen", unexpected_request)

    result = cli.main(
        ["export", "--remote", "--archive", str(tmp_path / "logbook.jsonl")]
    )

    assert result == 1
    assert "must use HTTPS" in capsys.readouterr().err
