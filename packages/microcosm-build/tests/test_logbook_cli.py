"""CLI contracts for the append-only Logbook archive."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse

import pytest

from microcosm.build.logbook import (
    LOGBOOK_PROVENANCE_ROW_FIELDS,
    LogbookRow,
    load_logbook_file,
)

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
    pipeline: str = "uk-frs-staging",
    rung: str = "f010",
    disposition: str = "failed",
    run_provenance_identity: dict[str, object] | None = None,
) -> LogbookRow:
    artifact = (
        f"hf://datasets/policyengine/populace-us@{build_id}"
        if disposition in {"published", "certified"}
        else None
    )
    return LogbookRow.create(
        build_id=build_id,
        ts=f"2026-08-05T12:{minute:02d}:00Z",
        pipeline=pipeline,
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
        run_provenance_identity=run_provenance_identity,
    )


def _run_provenance_identity() -> dict[str, object]:
    return {
        "identity_generation": 1,
        "source_grammar_receipt": {
            "schema_version": 3,
            "canonicalizer_version": 1,
            "migration_chain": [{"id": "fixture-v2-v3", "sha256": "3" * 64}],
        },
        "spec_binding": {
            "country": "us",
            "schema_id": "country-spec",
            "schema_version": 3,
            "canonicalizer_version": 1,
            "spec_sha256": "4" * 64,
            "attestation": "bundle-authoritative",
        },
        "authority_versions": {"stacked_authority": 10},
        "code_inventory_digest": "5" * 64,
        "artifact_protocol_inventory": {"parquet": "fixture-v1"},
        "run_request": {"config_authority": "bundle", "rung": "f004"},
        "execution_receipt": {"resolved_backend": "cpu"},
    }


def _chain(
    *,
    pipeline: str = "uk-frs-staging",
) -> tuple[LogbookRow, LogbookRow, LogbookRow]:
    first = _row(
        "fixture-build-1",
        predecessor=None,
        minute=1,
        pipeline=pipeline,
    )
    second = _row(
        "fixture-build-2",
        predecessor=first.row_digest,
        minute=2,
        pipeline=pipeline,
        rung="f100",
        disposition="published",
    )
    third = _row(
        "fixture-build-3",
        predecessor=second.row_digest,
        minute=3,
        pipeline=pipeline,
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

    assert cli.DEFAULT_ARCHIVE_ROOT == ROOT / "logbook"
    assert cli.DEFAULT_SPOOL_ROOT == ROOT / "logbook-spool"


def test_committed_archives_are_scoped_by_country() -> None:
    # One chain per country: the US pool lineage and the UK migration
    # chain never share an archive (microcosm#665).
    archives = sorted(
        path.relative_to(ROOT / "logbook").as_posix()
        for path in (ROOT / "logbook").rglob("*.jsonl")
    )
    assert archives == ["us.jsonl"]
    assert (ROOT / "logbook" / "README.md").is_file()


def test_render_sections_each_scope_separately(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Independent chains render as separate sections; merging them into one
    # table would imply an ordering across scopes the archives never assert.
    us = tmp_path / "us.jsonl"
    uk = tmp_path / "uk.jsonl"
    _write_jsonl(us, _chain())
    _write_jsonl(uk, _chain())
    cli = _load_cli()

    assert cli.main(["render", "--archive", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "## us" in out
    assert "## uk" in out


def test_validate_walks_every_scope_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    us = tmp_path / "us.jsonl"
    uk = tmp_path / "uk.jsonl"
    _write_jsonl(us, _chain())
    _write_jsonl(uk, _chain())
    cli = _load_cli()

    assert cli.main(["validate", "--archive", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "validated 3 Logbook rows in us;" in out
    assert "validated 3 Logbook rows in uk;" in out


def test_export_refuses_a_directory_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An export extends exactly one chain, so the scope must be named.
    spool = tmp_path / "spool"
    spool.mkdir()
    cli = _load_cli()

    assert (
        cli.main(["export", "--archive", str(tmp_path), "--source", str(spool)]) == 1
    )
    assert "extends exactly one scope chain" in capsys.readouterr().err


def test_export_requires_a_named_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()

    assert cli.main(["export", "--archive", str(tmp_path / "uk.jsonl")]) == 1
    assert "needs --source" in capsys.readouterr().err


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
    archive = tmp_path / "logbook" / "uk" / "frs.jsonl"
    archive.parent.mkdir(parents=True)
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


def test_cli_local_export_refuses_wrong_scope_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The adversarial finding on this PR: the chain verifier authenticates
    # payloads, never filenames, so without this gate a uk-locals spool
    # exported into the uk/frs archive would chain validly and
    # permanently mis-scope lineage. Local exports get the same scope
    # discipline as the remote branch.
    archive = tmp_path / "logbook" / "uk" / "frs.jsonl"
    archive.parent.mkdir(parents=True)
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, _chain(pipeline="uk-locals-rowwise"))
    cli = _load_cli()

    exit_code = cli.main(
        ["export", "--archive", str(archive), "--source", str(source)]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "outside scope uk/frs" in err
    assert "uk-locals-rowwise" in err
    assert not archive.exists()


def test_cli_export_refuses_an_unratified_scope_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # uk/firms derives cleanly but is not in the ratified vocabulary yet:
    # opening a scope is a reviewed diff (migration + CLI mirror + README),
    # never a side effect of a well-formed archive path.
    archive = tmp_path / "logbook" / "uk" / "firms.jsonl"
    archive.parent.mkdir(parents=True)
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, _chain(pipeline="uk-firms-staging"))
    cli = _load_cli()

    exit_code = cli.main(
        ["export", "--archive", str(archive), "--source", str(source)]
    )

    assert exit_code == 1
    assert "not in the ratified scope list" in capsys.readouterr().err
    assert not archive.exists()


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
    archive = tmp_path / "logbook" / "uk" / "frs.jsonl"
    archive.parent.mkdir(parents=True)
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
    archive = tmp_path / "logbook" / "uk" / "frs.jsonl"
    archive.parent.mkdir(parents=True)
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
        database_rows = []
        for row in rows:
            mapping = row.to_mapping()
            mapping.setdefault("run_provenance_identity", None)
            database_rows.append(mapping)
        self._payload = json.dumps(database_rows).encode()
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
    rows = _chain(pipeline="us-stacked-pool")
    archive = tmp_path / "logbook" / "us.jsonl"
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
    assert set(query["select"][0].split(",")) == LOGBOOK_PROVENANCE_ROW_FIELDS
    assert query["pipeline"] == [
        'in.("us-2024-release","us-pool-inc2","us-stacked-pool")'
    ]


def test_cli_remote_export_preserves_historical_null_and_f1_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    legacy = _row(
        "fixture-build-legacy",
        predecessor=None,
        minute=1,
        pipeline="us-stacked-pool",
    )
    f1 = _row(
        "fixture-build-f1",
        predecessor=legacy.row_digest,
        minute=2,
        pipeline="us-stacked-pool",
        rung="f004",
        run_provenance_identity=_run_provenance_identity(),
    )
    archive = tmp_path / "logbook" / "us.jsonl"
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda *_args, **_kwargs: _RemoteResponse((legacy, f1)),
    )

    assert cli.main(["export", "--remote", "--archive", str(archive)]) == 0

    restored = load_logbook_file(archive)
    assert restored == (legacy, f1)
    assert restored[0].row_digest == legacy.row_digest
    assert "run_provenance_identity" not in restored[0].to_mapping()
    assert restored[1].run_provenance_identity == _run_provenance_identity()


def test_cli_remote_export_filters_nested_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _chain(pipeline="uk-frs-staging")
    archive = tmp_path / "logbook" / "uk" / "frs.jsonl"
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: float) -> _RemoteResponse:
        del timeout
        requests.append(request)
        return _RemoteResponse(rows)

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    assert cli.main(["export", "--remote", "--archive", str(archive)]) == 0

    assert load_logbook_file(archive) == rows
    query = parse_qs(urlparse(requests[0].full_url).query)
    assert set(query["select"][0].split(",")) == LOGBOOK_PROVENANCE_ROW_FIELDS
    assert query["pipeline"] == ["like.uk-frs-*"]


def test_remote_normalization_only_removes_the_historical_null() -> None:
    cli = _load_cli()
    historical = {"build_id": "legacy", "run_provenance_identity": None}
    binding = _run_provenance_identity()
    current = {"build_id": "f1", "run_provenance_identity": binding}

    assert cli._normalize_remote_row(historical) == {"build_id": "legacy"}
    assert cli._normalize_remote_row(current) == current
    assert current["run_provenance_identity"] is binding


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
        [
            "export",
            "--remote",
            "--archive",
            str(tmp_path / "logbook" / "us.jsonl"),
        ]
    )

    assert result == 1
    assert "POPULACE_LEDGER_EXPORT_KEY" in capsys.readouterr().err


def test_cli_remote_export_paginates_before_chain_ordering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _chain(pipeline="us-stacked-pool")
    archive = tmp_path / "logbook" / "us.jsonl"
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
        [
            "export",
            "--remote",
            "--archive",
            str(tmp_path / "logbook" / "us.jsonl"),
        ]
    )

    assert result == 1
    assert "must use HTTPS" in capsys.readouterr().err


def test_cli_remote_export_refuses_unscoped_archive_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")

    def unexpected_request(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unscoped archive must fail before a request")

    monkeypatch.setattr(cli, "urlopen", unexpected_request)

    result = cli.main(
        ["export", "--remote", "--archive", str(tmp_path / "logbook.jsonl")]
    )

    assert result == 1
    assert "must be logbook/us.jsonl" in capsys.readouterr().err


def test_cli_remote_export_refuses_wrong_scope_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _chain(pipeline="uk-locals-rowwise")
    archive = tmp_path / "logbook" / "uk" / "frs.jsonl"
    cli = _load_cli()
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_EXPORT_KEY", "exporter-jwt")
    monkeypatch.setenv("POPULACE_LEDGER_API_KEY", "project-api-key")
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda *_args, **_kwargs: _RemoteResponse(rows),
    )

    result = cli.main(["export", "--remote", "--archive", str(archive)])

    assert result == 1
    err = capsys.readouterr().err
    assert "outside scope uk/frs" in err
    assert "uk-locals-rowwise" in err


@pytest.mark.parametrize(
    ("pipeline", "scope"),
    [
        ("us-2024-release", "us"),
        ("us-pool-inc2", "us"),
        ("us-stacked-pool", "us"),
        ("uk-frs-staging", "uk/frs"),
        ("uk-locals-rowwise", "uk/locals"),
        ("us-pool-inc3", "us/pool"),
        ("mystery-pipeline", None),
    ],
)
def test_chain_scope_matches_sql_contract(
    pipeline: str,
    scope: str | None,
) -> None:
    cli = _load_cli()

    assert cli._chain_scope(pipeline) == scope
