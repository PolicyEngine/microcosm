"""Tests for the Census imports ingest of the US import-entry unit family."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.us_runtime.us_trade.census_country_bridge import (
    BRIDGE_SHA256,
    load_census_country_bridge,
)
from microcosm.build.us_runtime.us_trade.census_imports import (
    CensusImportsMonth,
    _elide_key,
    _reconcile_against_census_totals,
    assemble_margins_table,
    fetch_imports_month,
    month_range,
    parse_imports_response,
)

GOLDEN = Path(__file__).parent / "golden" / "us_trade"
LINE_FIXTURE = GOLDEN / "census_imports_hs10_2026-03_line.json"


def _fixture_bytes() -> bytes:
    return LINE_FIXTURE.read_bytes()


def test_bridge_loads_and_is_fail_closed():
    bridge = load_census_country_bridge()
    assert len(bridge) == 241
    assert bridge.sha256 == BRIDGE_SHA256
    assert bridge.iso2("1220") == "CA"
    assert bridge.iso2("2010") == "MX"
    with pytest.raises(KeyError, match="fail-closed"):
        bridge.iso2("9999")


def test_bridge_provenance_is_vendored_pinned_and_cross_bound():
    """The vendored provenance record must hash to its pin, declare exactly
    the vendored CSV's hash and row count, and the upstream URL must cite an
    immutable commit rather than a branch."""

    import hashlib
    import json
    import re
    from importlib.resources import files

    from microcosm.build.us_runtime.us_trade.census_country_bridge import (
        BRIDGE_PROVENANCE_RESOURCE_NAME,
        BRIDGE_PROVENANCE_SHA256,
        BRIDGE_UPSTREAM,
        BRIDGE_UPSTREAM_COMMIT,
    )

    raw = (
        files("microcosm.build.us_runtime.us_trade")
        .joinpath(BRIDGE_PROVENANCE_RESOURCE_NAME)
        .read_bytes()
    )
    assert hashlib.sha256(raw).hexdigest() == BRIDGE_PROVENANCE_SHA256
    provenance = json.loads(raw)
    assert provenance["bridge_sha256"] == BRIDGE_SHA256
    assert provenance["bridge_rows"] == len(load_census_country_bridge())
    assert provenance["snapshot_sha256"]
    assert re.fullmatch(r"[0-9a-f]{40}", BRIDGE_UPSTREAM_COMMIT)
    assert BRIDGE_UPSTREAM_COMMIT in BRIDGE_UPSTREAM
    assert "/blob/main/" not in BRIDGE_UPSTREAM


def test_parse_real_response_splits_detail_and_totals():
    countries, totals = parse_imports_response(_fixture_bytes(), "2026-03", "31")
    assert len(totals) == 1
    assert len(countries) == 3
    assert {row["hts10"] for row in countries} == {"3103110000"}
    assert all(row["period"] == "2026-03" for row in countries + totals)
    assert totals[0]["cty_code"] == "-"
    for measure in ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo"):
        assert isinstance(totals[0][measure], int)


def test_parse_rejects_wrong_month_and_prefix():
    with pytest.raises(ValueError, match="echoes month"):
        parse_imports_response(_fixture_bytes(), "2026-04", "31")
    with pytest.raises(ValueError, match="prefix 32"):
        parse_imports_response(_fixture_bytes(), "2026-03", "32")
    with pytest.raises(ValueError, match="prefix 3104"):
        parse_imports_response(_fixture_bytes(), "2026-03", "3104")


def test_parse_rejects_unexpected_detail_country_code():
    table = json.loads(_fixture_bytes().decode())
    header = table[0]
    row = list(table[1])
    row[header.index("CTY_CODE")] = "0099"
    row[header.index("SUMMARY_LVL")] = "DET"
    doctored = json.dumps([header, row]).encode()
    with pytest.raises(ValueError, match="unexpected country code"):
        parse_imports_response(doctored, "2026-03", "31")


def test_reconciliation_is_exact_on_publisher_totals():
    countries, totals = parse_imports_response(_fixture_bytes(), "2026-03", "31")
    assert _reconcile_against_census_totals(countries, totals, "2026-03") == ()
    tampered = [dict(row) for row in countries]
    tampered[0]["con_val_mo"] = int(tampered[0]["con_val_mo"]) + 1
    failures = _reconcile_against_census_totals(tampered, totals, "2026-03")
    assert len(failures) == 1
    assert "con_val_mo" in failures[0]


def test_reconciliation_gates_active_detail_without_published_total():
    """Coverage is part of the reconciliation: iterating only published
    total keys let active detail pass unreconciled whenever its '-' row
    was missing. Active detail without a total must fail; a zero-carrier
    detail row without a total stays agreement-by-absence."""

    countries, _ = parse_imports_response(_fixture_bytes(), "2026-03", "31")
    failures = _reconcile_against_census_totals(countries, [], "2026-03")
    assert len(failures) == 1
    assert "no published '-' total row" in failures[0]
    assert "3103110000" in failures[0]

    zero_carrier = dict(countries[0])
    for measure in ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo"):
        zero_carrier[measure] = 0
    for measure in ("con_qy1_mo", "gen_qy1_mo"):
        zero_carrier[measure] = 0
    assert _reconcile_against_census_totals([zero_carrier], [], "2026-03") == ()

    quantity_only = dict(zero_carrier)
    quantity_only["gen_qy1_mo"] = 5
    failures = _reconcile_against_census_totals([quantity_only], [], "2026-03")
    assert len(failures) == 1
    assert "no published '-' total row" in failures[0]


def test_assemble_bridges_fail_closed_and_types_margins():
    bridge = load_census_country_bridge()
    countries, totals = parse_imports_response(_fixture_bytes(), "2026-03", "31")
    month = CensusImportsMonth(
        month="2026-03",
        country_rows=tuple(countries),
        total_rows=tuple(totals),
        manifest_entries=(),
        reconciliation_failures=(),
    )
    pull = assemble_margins_table((month,), bridge)
    assert list(pull.margins["chapter"].unique()) == ["31"]
    assert pull.margins["con_val_mo"].dtype == "int64"
    assert pull.margins["iso2"].notna().all()
    assert len(pull.margins) == 3
    assert len(pull.census_totals) == 1
    summed = int(pull.margins["con_val_mo"].sum())
    assert summed == int(pull.census_totals["con_val_mo"].iloc[0])

    unbridged = dict(countries[0])
    unbridged["cty_code"] = "5999"
    broken = CensusImportsMonth(
        month="2026-03",
        country_rows=(unbridged,),
        total_rows=(),
        manifest_entries=(),
        reconciliation_failures=(),
    )
    with pytest.raises(KeyError, match="fail-closed"):
        assemble_margins_table((broken,), bridge)


def test_fetch_caches_bytes_and_manifest(tmp_path):
    calls: list[str] = []

    def fake_fetch(url: str):
        calls.append(url)
        if "I_COMMODITY=31%2A" in url or "I_COMMODITY=31*" in url:
            return 200, _fixture_bytes()
        return 204, b""

    month = fetch_imports_month(
        "2026-03",
        "test-key",
        cache_dir=tmp_path,
        chapters=("31", "77"),
        fetch=fake_fetch,
        throttle_seconds=0.0,
    )
    assert len(calls) == 2
    assert len(month.country_rows) == 3
    assert month.reconciliation_failures == ()
    entries = {entry["chapter"]: entry for entry in month.manifest_entries}
    assert entries["31"]["http_status"] == 200
    assert entries["31"]["row_count"] == 12
    assert entries["31"]["sha256"] == hashlib.sha256(_fixture_bytes()).hexdigest()
    assert entries["77"]["http_status"] == 204
    assert entries["77"]["row_count"] == 0
    for entry in month.manifest_entries:
        assert "key=" not in str(entry["url"])
        assert entry["retrieved_at"]

    cached = fetch_imports_month(
        "2026-03",
        "test-key",
        cache_dir=tmp_path,
        chapters=("31", "77"),
        fetch=fake_fetch,
        throttle_seconds=0.0,
    )
    assert len(calls) == 2
    assert cached.country_rows == month.country_rows

    data_file = tmp_path / "imports_hs10_2026-03_ch31.json"
    data_file.write_bytes(data_file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match its recorded hash"):
        fetch_imports_month(
            "2026-03",
            "test-key",
            cache_dir=tmp_path,
            chapters=("31",),
            fetch=fake_fetch,
            throttle_seconds=0.0,
        )


def test_overloaded_chapter_splits_into_prefixes_and_resumes(tmp_path):
    """A chapter the server cannot materialize fans out to 3-digit prefixes."""
    calls: list[str] = []

    def fake_fetch(url: str):
        calls.append(url)
        commodity = url.split("I_COMMODITY=")[1].split("&")[0]
        if commodity in ("31%2A", "31*"):
            return 500, b""
        if commodity in ("310%2A", "310*"):
            return 200, _fixture_bytes()
        return 204, b""

    month = fetch_imports_month(
        "2026-03",
        "test-key",
        cache_dir=tmp_path,
        chapters=("31",),
        fetch=fake_fetch,
        throttle_seconds=0.0,
    )
    assert len(calls) == 11  # the doomed chapter probe + ten sub-prefixes
    assert len(month.country_rows) == 3
    assert month.reconciliation_failures == ()
    by_prefix = {entry["prefix"]: entry for entry in month.manifest_entries}
    assert set(by_prefix) == {"31"} | {f"31{digit}" for digit in "0123456789"}
    assert by_prefix["31"]["superseded_by_split"] is True
    assert by_prefix["31"]["sha256"] is None
    assert by_prefix["310"]["http_status"] == 200
    assert by_prefix["310"]["filename"] == "imports_hs10_2026-03_p310.json"
    assert by_prefix["314"]["http_status"] == 204

    resumed = fetch_imports_month(
        "2026-03",
        "test-key",
        cache_dir=tmp_path,
        chapters=("31",),
        fetch=fake_fetch,
        throttle_seconds=0.0,
    )
    assert len(calls) == 11  # marker + caches make the resume network-free
    assert resumed.country_rows == month.country_rows


def test_month_range_and_key_elision():
    assert month_range("2025-11", "2026-02") == (
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    )
    with pytest.raises(ValueError, match="after end"):
        month_range("2026-02", "2026-01")
    url = "https://api.census.gov/data/x?get=A,B&time=2026-01&key=SECRET"
    elided = _elide_key(url)
    assert "SECRET" not in elided
    assert "time=2026-01" in elided


def test_cache_reuse_preserves_marker_provenance_and_needs_no_writes(
    tmp_path, monkeypatch
):
    """Reuse is not retrieval: resuming over a warm split-tree cache must
    reuse the split marker verbatim — original retrieved_at — and issue
    zero write calls. The spy intercepts every write primitive the module
    can reach (Path.write_text / Path.write_bytes / writable open), which
    a directory-permission probe cannot prove: a read-only *directory*
    still lets 0644 sidecars be rewritten in place, and a rewrite within
    the marker's one-second timestamp precision reproduces identical
    bytes, passing any byte-snapshot comparison."""

    def fake_fetch(url: str):
        commodity = url.split("I_COMMODITY=")[1].split("&")[0]
        if commodity in ("31%2A", "31*"):
            return 500, b""
        if commodity in ("310%2A", "310*"):
            return 200, _fixture_bytes()
        return 204, b""

    warm = fetch_imports_month(
        "2026-03",
        "test-key",
        cache_dir=tmp_path,
        chapters=("31",),
        fetch=fake_fetch,
        throttle_seconds=0.0,
    )
    warm_marker = next(
        entry for entry in warm.manifest_entries if entry.get("superseded_by_split")
    )
    snapshot = {path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())}

    def refusing_fetch(url: str):
        raise AssertionError(f"resume must not fetch, asked for {url}")

    writes: list[str] = []
    real_open = builtins.open

    def spy_write_text(self, data, *args, **kwargs):
        writes.append(f"write_text:{self}")
        return len(data)

    def spy_write_bytes(self, data, *args, **kwargs):
        writes.append(f"write_bytes:{self}")
        return len(data)

    def spy_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            writes.append(f"open[{mode}]:{file}")
            raise AssertionError(f"resume opened {file!r} writable ({mode!r})")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(Path, "write_bytes", spy_write_bytes)
    monkeypatch.setattr(builtins, "open", spy_open)
    resumed = fetch_imports_month(
        "2026-03",
        "test-key",
        cache_dir=tmp_path,
        chapters=("31",),
        fetch=refusing_fetch,
        throttle_seconds=0.0,
    )
    monkeypatch.undo()

    assert writes == []
    assert resumed.country_rows == warm.country_rows
    resumed_marker = next(
        entry for entry in resumed.manifest_entries if entry.get("superseded_by_split")
    )
    assert resumed_marker["retrieved_at"] == warm_marker["retrieved_at"]
    assert {
        path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())
    } == snapshot
