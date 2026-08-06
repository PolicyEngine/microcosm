"""CLI contracts for the append-only Chronicle archive."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from populace.build.chronicle import ChronicleRow, load_chronicle_file

ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = ROOT / "tools/chronicle.py"


def _row(
    build_id: str,
    *,
    predecessor: str | None,
    minute: int,
    rung: str = "f010",
    disposition: str = "failed",
) -> ChronicleRow:
    artifact = (
        f"hf://datasets/policyengine/populace-us@{build_id}"
        if disposition in {"published", "certified"}
        else None
    )
    return ChronicleRow.create(
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


def _chain() -> tuple[ChronicleRow, ChronicleRow, ChronicleRow]:
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


def _write_jsonl(path: Path, rows: tuple[ChronicleRow, ...]) -> None:
    path.write_text("".join(row.to_json_line() for row in rows), encoding="utf-8")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("chronicle_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_the_ratified_root_paths() -> None:
    cli = _load_cli()

    assert cli.DEFAULT_ARCHIVE == ROOT / "chronicle.jsonl"
    assert cli.DEFAULT_SOURCE == ROOT / "ledger-spool"


def test_cli_validate_and_filtered_render(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "chronicle.jsonl"
    _write_jsonl(archive, _chain())
    cli = _load_cli()

    assert cli.main(["validate", "--archive", str(archive)]) == 0
    assert "validated 3 Chronicle rows" in capsys.readouterr().out

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
    archive = tmp_path / "chronicle.jsonl"
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
    assert "exported 2 new Chronicle rows" in capsys.readouterr().out
    assert load_chronicle_file(archive) == (first, second, third)

    assert cli.main(command) == 0
    assert "exported 0 new Chronicle rows" in capsys.readouterr().out
    assert load_chronicle_file(archive) == (first, second, third)


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
    archive = tmp_path / "chronicle.jsonl"
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

    assert "exported 3 new Chronicle rows" in capsys.readouterr().out
    assert load_chronicle_file(archive) == rows


def test_cli_export_divergence_fails_closed_without_modifying_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first, _, _ = _chain()
    archive = tmp_path / "chronicle.jsonl"
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

    assert "chronicle export failed" in capsys.readouterr().err
    assert archive.read_bytes() == before
