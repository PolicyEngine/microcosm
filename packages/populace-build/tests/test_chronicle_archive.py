"""Contracts for Chronicle's checked-in git archive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import populace.build.chronicle as chronicle
from populace.build.chronicle import (
    ChronicleRow,
    export_rows,
    load_chronicle_file,
    render_markdown,
)


def _row(
    build_id: str,
    *,
    predecessor: str | None,
    disposition: str = "failed",
    artifact: str | None = None,
) -> ChronicleRow:
    return ChronicleRow.create(
        build_id=build_id,
        ts="2026-08-05T12:00:00Z",
        pipeline="archive-fixture",
        rung="f100",
        seed=0,
        code_pin="fixture",
        input_pins_digest="1" * 64,
        identity_digest="2" * 64,
        phases_reached=["assembled"],
        gate_verdicts={
            "terminal": {
                "verdict": disposition,
                "receipt": "receipt://archive-fixture",
            }
        },
        wall_seconds=1,
        cost_usd=2,
        artifact_location=artifact,
        disposition=disposition,
        prediction_id=None,
        prev_row_digest=predecessor,
    )


def test_export_appends_only_a_contiguous_suffix(tmp_path: Path) -> None:
    archive = tmp_path / "chronicle.jsonl"
    first = _row("archive-1", predecessor=None)
    second = _row("archive-2", predecessor=first.row_digest)

    initial = export_rows(archive, [first])
    appended = export_rows(archive, [second])
    replay = export_rows(archive, [first, second])

    assert (initial.existing, initial.appended) == (0, 1)
    assert (appended.existing, appended.appended) == (1, 1)
    assert (replay.existing, replay.appended) == (2, 0)
    assert load_chronicle_file(archive) == (first, second)


def test_export_fails_closed_when_suffix_does_not_extend_tail(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "chronicle.jsonl"
    first = _row("archive-1", predecessor=None)
    export_rows(archive, [first])
    original = archive.read_bytes()
    divergent = _row("archive-2", predecessor="f" * 64)

    with pytest.raises(ValueError, match=r"diverges.*archive tail"):
        export_rows(archive, [divergent])

    assert archive.read_bytes() == original


def test_export_reauthenticates_mutable_nested_row_state(tmp_path: Path) -> None:
    archive = tmp_path / "chronicle.jsonl"
    row = _row("archive-1", predecessor=None)
    row.gate_verdicts["terminal"]["verdict"] = "tampered"

    with pytest.raises(ValueError, match=r"row_digest.*does not match"):
        export_rows(archive, [row])

    assert not archive.exists()


def test_export_retry_completes_parent_fsync_after_interrupted_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "chronicle.jsonl"
    row = _row("archive-1", predecessor=None)
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
        export_rows(archive, [row])
    assert load_chronicle_file(archive) == (row,)

    retry = export_rows(archive, [row])

    assert (retry.existing, retry.appended) == (1, 0)
    assert parent_fsync_calls == 2


def test_mid_row_tamper_names_the_row(tmp_path: Path) -> None:
    archive = tmp_path / "chronicle.jsonl"
    first = _row("archive-1", predecessor=None)
    second = _row("archive-2", predecessor=first.row_digest)
    third = _row("archive-3", predecessor=second.row_digest)
    export_rows(archive, [first, second, third])
    lines = archive.read_text(encoding="utf-8").splitlines()
    mutated = json.loads(lines[1])
    mutated["pipeline"] = "tampered"
    lines[1] = json.dumps(mutated, separators=(",", ":"), sort_keys=True)
    archive.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"row 2 \(archive-2\).*row_digest"):
        load_chronicle_file(archive)


def test_render_filters_and_uses_public_safe_projection() -> None:
    failed = _row("archive-failed", predecessor=None)
    published = _row(
        "archive-published",
        predecessor=failed.row_digest,
        disposition="published",
        artifact="hf-tag",
    )

    table = render_markdown(
        [failed, published],
        rung="f100",
        dispositions={"published"},
    )

    assert "archive-published" in table
    assert "archive-failed" not in table
    assert "cost_usd" not in table
    assert "gate_verdicts" not in table
