"""Contracts for the direct #628-compatible Chronicle spool."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import populace.build.chronicle as chronicle
from populace.build.chronicle import (
    CHRONICLE_ROW_FIELDS,
    CHRONICLE_RUNGS,
    ChronicleRow,
    canonical_json_bytes,
    load_chronicle_row,
    record_build_attempt,
)


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


def test_canonical_json_matches_sql_number_and_unicode_vector() -> None:
    value = {"z": 1.0, "a": ["é", 1e-3, None], "n": -0.0}

    assert canonical_json_bytes(value).decode() == (
        '{"a":["é",0.001,null],"n":0,"z":1}'
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("build_id", "bad build id", "build_id"),
        ("ts", "2026-08-04T19:06:00", "UTC offset"),
        ("pipeline", " fixture ", "pipeline"),
        ("rung", "1/10", "rung must be one of"),
        ("rung", "0.1", "rung must be one of"),
        ("rung", "f020", "rung must be one of"),
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


def test_wrong_supplied_row_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="row_digest"):
        ChronicleRow.create(**_row_kwargs(), row_digest="f" * 64)


def test_published_row_requires_an_artifact_location() -> None:
    with pytest.raises(ValueError, match="artifact_location is required"):
        ChronicleRow.create(**_row_kwargs(disposition="published"))


@pytest.mark.parametrize("rung", sorted(CHRONICLE_RUNGS))
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


def test_identical_retry_is_idempotent(tmp_path: Path) -> None:
    first = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)
    original_bytes = first.spool_path.read_bytes()

    second = record_build_attempt(**_row_kwargs(), spool_dir=tmp_path)

    assert second == first
    assert second.spool_path.read_bytes() == original_bytes
    assert len(list(tmp_path.glob("*.json"))) == 1
