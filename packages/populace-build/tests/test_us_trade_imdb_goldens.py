"""Archive-fidelity goldens for the IMDB bulk ingest (populace#615 P1).

The committed golden pack under ``golden/us_trade/imdb/`` carries, for two
real archives — IMDB2501 (2025-01) and IMDB2606 (2026-06) — the official
record layouts (``*.lay``, verbatim bytes from each archive's
``Documentation/`` directory), a selection of verbatim raw fixed-width
records, the production parse of those records as literal expected values,
and a manifest binding everything to the source archives' sha256s.

Together these make archive-level parse fidelity reviewable in-branch:

- the layout test machine-checks every hard-coded production field span
  against the official ``.lay`` bytes, so a transcription error — even one
  applied consistently across detail and controls — cannot survive;
- the raw-record test runs the committed official lines through the real
  production parser and compares every field to committed literals that a
  reviewer can verify by hand against the ``.lay`` positions;
- when the source archives are present on disk, the linkage test recomputes
  their sha256s against the manifest and re-extracts the committed lines
  byte-for-byte.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import pytest

from populace.build.us_runtime.us_trade.imdb_bulk import (
    _COMM_FIELDS,
    _CTY_FIELDS,
    _DE_FIELDS,
    _DETAIL_KEYS,
    _DETAIL_MEASURES,
    _read_fixed_width,
)

GOLDEN_DIR = Path(__file__).parent / "golden" / "us_trade" / "imdb"
#: Where the full source archives live when cached locally; the linkage
#: test skips cleanly when they are absent (CI does not carry 100+ MB
#: archives — the golden manifest's sha chain still binds the pack).
ARCHIVE_DIR = Path(
    os.environ.get(
        "POPULACE_IMDB_ARCHIVE_DIR",
        str(Path.home() / "PolicyEngine" / "_laneG-runtime" / "bulk" / "imdb"),
    )
)
ARCHIVE_STEMS = ("IMDB2501", "IMDB2606")

_PRODUCTION_FIELDS = {
    "IMP_DETL": _DETAIL_KEYS + _DETAIL_MEASURES,
    "IMP_CTY": _CTY_FIELDS,
    "IMP_COMM": _COMM_FIELDS,
    "IMP_DE": _DE_FIELDS,
}
_MEASURE_NAMES = {
    kind: [name for name, _, _ in fields if name.endswith("_mo")]
    for kind, fields in _PRODUCTION_FIELDS.items()
}
#: Production names one field differently from the official layouts: the
#: layouts call the 10-digit HTS code ``commodity``.
_LAY_NAME_FOR = {"hts10": "commodity"}

_LAY_FIELD_PATTERN = re.compile(r"^\s*(\d+)-(\d+)\s+(\S+)", re.MULTILINE)


def _parse_lay(text: str) -> dict[str, tuple[int, int]]:
    """Official layout text -> {field name: (1-indexed start, inclusive end)}."""
    fields: dict[str, tuple[int, int]] = {}
    for match in _LAY_FIELD_PATTERN.finditer(text):
        name = match.group(3)
        span = (int(match.group(1)), int(match.group(2)))
        assert name not in fields, f"duplicate .lay field {name}"
        fields[name] = span
    return fields


def _manifest() -> dict:
    return json.loads((GOLDEN_DIR / "golden_manifest.json").read_text())


@pytest.mark.parametrize("stem", ARCHIVE_STEMS)
@pytest.mark.parametrize("kind", sorted(_PRODUCTION_FIELDS))
def test_production_layouts_match_official_lay_files(stem, kind):
    """Every hard-coded production span equals the official .lay span.

    The .lay bytes are committed verbatim from the archives themselves (and
    sha-bound to the archives in the golden manifest), so this check makes a
    correlated transcription error across detail and control layouts
    impossible to hold silently.
    """
    lay = _parse_lay((GOLDEN_DIR / stem / f"{kind}.lay").read_text("latin-1"))
    for name, start, end in _PRODUCTION_FIELDS[kind]:
        lay_name = _LAY_NAME_FOR.get(name, name)
        assert lay_name in lay, f"{stem}/{kind}: {lay_name} missing from .lay"
        assert lay[lay_name] == (start, end), (
            f"{stem}/{kind}.{lay_name}: production parses {start}-{end}, "
            f"official layout says {lay[lay_name][0]}-{lay[lay_name][1]}"
        )


@pytest.mark.parametrize("stem", ARCHIVE_STEMS)
def test_raw_record_goldens_parse_via_production_parser(stem):
    """Committed official raw lines, parsed by the real production parser,
    equal the committed literal expected values field for field."""

    raw_records = [
        json.loads(line)
        for line in (GOLDEN_DIR / stem / "raw_records.jsonl").read_text().splitlines()
    ]
    expected_records = {
        (record["member_kind"], record["line"]): record["fields"]
        for record in json.loads(
            (GOLDEN_DIR / stem / "expected_values.json").read_text()
        )
    }
    assert len(raw_records) == len(expected_records)
    by_kind: dict[str, list[dict]] = {}
    for record in raw_records:
        by_kind.setdefault(record["member_kind"], []).append(record)
    assert set(by_kind) == set(_PRODUCTION_FIELDS)
    for kind, records in by_kind.items():
        member_name = records[0]["member"]
        payload = ("\r\n".join(record["raw"] for record in records) + "\r\n").encode(
            "latin-1"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr(member_name, payload)
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as bundle:
            frame = _read_fixed_width(
                bundle,
                member_name,
                _PRODUCTION_FIELDS[kind],
                measure_names=list(_MEASURE_NAMES[kind]),
            )
        assert len(frame) == len(records)
        for position, record in enumerate(records):
            expected = expected_records[(kind, record["line"])]
            parsed = frame.iloc[position]
            for field_name, expected_value in expected.items():
                actual = parsed[field_name]
                if field_name in _MEASURE_NAMES[kind]:
                    assert int(actual) == expected_value, (
                        f"{stem}/{kind} line {record['line']} {field_name}"
                    )
                else:
                    assert actual == expected_value, (
                        f"{stem}/{kind} line {record['line']} {field_name}"
                    )


def test_golden_pack_is_internally_bound():
    """The manifest lists both months with all four members, and the
    committed .lay bytes hash to the manifest's recorded sha256s."""

    manifest = _manifest()
    assert set(manifest["archives"]) == set(ARCHIVE_STEMS)
    for stem, archive in manifest["archives"].items():
        assert set(archive["members"]) == set(_PRODUCTION_FIELDS)
        assert set(archive["lay_members"]) == set(_PRODUCTION_FIELDS)
        assert re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
        for kind, lay_entry in archive["lay_members"].items():
            committed = (GOLDEN_DIR / stem / f"{kind}.lay").read_bytes()
            assert hashlib.sha256(committed).hexdigest() == lay_entry["sha256"], (
                f"{stem}/{kind}.lay does not hash to its manifest sha"
            )
        for kind, member_entry in archive["members"].items():
            selected = set(member_entry["selected_lines"])
            raw_lines = {
                record["line"]
                for record in (
                    json.loads(line)
                    for line in (GOLDEN_DIR / stem / "raw_records.jsonl")
                    .read_text()
                    .splitlines()
                )
                if record["member_kind"] == kind
            }
            assert raw_lines == selected


@pytest.mark.parametrize("stem", ARCHIVE_STEMS)
def test_goldens_match_source_archives_when_present(stem):
    """With the real archive on disk: its bytes hash to the manifest sha and
    re-extracting the selected lines reproduces the committed records
    byte-for-byte. (Skipped where the 100+ MB archives are not cached; the
    sha chain in the golden manifest still binds the committed pack.)"""

    archive_path = ARCHIVE_DIR / f"{stem}.ZIP"
    if not archive_path.is_file():
        pytest.skip(f"source archive {archive_path} not cached locally")
    manifest = _manifest()["archives"][stem]
    raw_zip = archive_path.read_bytes()
    assert hashlib.sha256(raw_zip).hexdigest() == manifest["sha256"]
    raw_records = [
        json.loads(line)
        for line in (GOLDEN_DIR / stem / "raw_records.jsonl").read_text().splitlines()
    ]
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as bundle:
        for kind, member_entry in manifest["members"].items():
            lines = bundle.read(member_entry["member"]).decode("latin-1").split("\r\n")
            for record in raw_records:
                if record["member_kind"] != kind:
                    continue
                assert lines[record["line"] - 1] == record["raw"], (
                    f"{stem}/{kind} line {record['line']} drifted from archive"
                )
        for kind, lay_entry in manifest["lay_members"].items():
            committed = (GOLDEN_DIR / stem / f"{kind}.lay").read_bytes()
            assert bundle.read(lay_entry["member"]) == committed
