"""End-to-end tests for the synthetic import-entry build CLI (#615 P2).

The fixture is a minimal but internally *authenticated* P1 publication:
margins parquet, archived CBP page, source manifest, and a build report
whose pins bind them together — the same chain the real P1 CLI publishes.
Covers the review-critical properties end to end:

- **Byte determinism**: two invocations over the same P1 publication
  produce byte-identical parquet, register, and validation report — no
  wall-clock value enters any output.
- **Input authentication**: the margins parquet and archived CBP page must
  hash to the P1 build report's pins (directly and through the source
  manifest); a substituted-but-parseable input is refused, not adopted.
- **Engine contract**: the generated columns satisfy the sha-pinned
  composition input-surface contract fixture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

GOLDEN = Path(__file__).parent / "golden" / "us_trade"
CLI = Path(__file__).resolve().parents[3] / "tools" / "build_us_import_entries.py"
CONTRACT = GOLDEN / "engine_entry_contract.json"


def _margins_dir(tmp_path: Path) -> Path:
    """A minimal but internally authenticated P1 publication directory."""
    margins_dir = tmp_path / "margins"
    margins_dir.mkdir()
    frame = pd.DataFrame(
        [
            ("2025-12", "0101210010", "01", "1220", "CA", 5_000),
            ("2026-01", "0101210010", "01", "1220", "CA", 90_000),
            ("2026-01", "8471300100", "84", "5700", "CN", 12_345_678),
            ("2026-02", "8471300100", "84", "5700", "CN", 7_654_321),
            ("2026-02", "9903810100", "99", "2010", "MX", 800),
            ("2026-02", "6109100012", "61", "5700", "CN", 0),
        ],
        columns=["period", "hts10", "chapter", "cty_code", "iso2", "con_val_mo"],
    )
    margins_path = margins_dir / "margins_hts10_country_month.parquet"
    frame.to_parquet(margins_path, index=False)
    page_bytes = (GOLDEN / "cbp_trade_stats_fragment.html").read_bytes()
    (margins_dir / "cbp_newsroom_stats_trade.html").write_bytes(page_bytes)
    manifest_path = margins_dir / "source_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "source_name": "cbp_trade_stats",
                "filename": "cbp_newsroom_stats_trade.html",
                "sha256": hashlib.sha256(page_bytes).hexdigest(),
                "retrieved_at": "2026-08-05T14:00:00+00:00",
            },
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "extracted_at": "2026-08-05T14:00:00+00:00",
        "facts_sha256": "ee" * 32,
        "margins_parquet_sha256": hashlib.sha256(margins_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (margins_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return margins_dir


def _run(margins_dir: Path, out_dir: Path, start: str = "2026-01"):
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--margins-dir",
            str(margins_dir),
            "--out-dir",
            str(out_dir),
            "--start",
            start,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_cli_builds_labeled_validated_entries(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    out_dir = tmp_path / "entries"
    result = _run(margins_dir, out_dir)
    assert result.returncode == 0, result.stderr

    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["exact"] is True
    assert report["cells_checked"] == 4  # 2025-12 outside window, zero cell skipped
    assert report["window"] == {"start": "2026-01", "end": "2026-02", "months": 2}
    assert report["engine_runnable_months"] == ["2026-02"]
    assert report["chapter_country_cells_exact"] == 4
    assert "generated_at" not in report
    assert report["p1_build"]["extracted_at"] == "2026-08-05T14:00:00+00:00"

    register = json.loads((out_dir / "assumptions.json").read_text())
    assert register["synthetic"] is True
    assert register["known_gaps"]["informal_share_proxy"]
    assert register["size_model"]["sigma_calibration_class"] == "proxy_moment_match"
    # The detached register authenticates its margins publication exactly
    # like the parquet metadata and the validation report do.
    assert register["p1_build"] == report["p1_build"]
    assert register["p1_build"]["margins_parquet_sha256"]
    assert register["p1_build"]["build_report_sha256"]
    # The CBP-anchored default build earns the CBP wording.
    assert "CBP's published national mean entry value" in register["statement"]
    assert "CBP's published informal-entry count share" in register["statement"]
    anchors = register["size_model"]["anchor_provenance"]
    assert anchors["fiscal_year"] == 2026
    assert anchors["mean_source"] == "cbp_published"
    assert anchors["share_source"] == "cbp_published"
    assert anchors["total_entry_summaries"] == 83_133_856
    assert 0.68 < anchors["informal_count_share"] < 0.69
    assert 30_000 < anchors["mean_entry_value"] < 36_000
    # The FY2026 cells cover October 2025 through the page's as-of date:
    # the anchors are fiscal-year-to-date window statistics and say so.
    assert anchors["period_coverage"] == "fiscal_year_to_date"
    assert anchors["coverage_as_of"] == "2026-07-27"
    assert "fiscal-year-to-date FY2026" in anchors["basis"]

    table = pq.read_table(out_dir / "synthetic_import_entries.parquet")
    metadata = table.schema.metadata
    assert metadata[b"populace_us_trade.synthetic"] == b"true"
    # The embedded register and the standalone file are the same bytes.
    assert metadata[b"populace_us_trade.assumptions"] == (
        (out_dir / "assumptions.json").read_bytes()
    )
    generator = json.loads(metadata[b"populace_us_trade.generator"])
    assert generator["issue"] == "PolicyEngine/microcosm#615"
    assert generator["synthetic"] is True
    assert "generated_at" not in generator
    assert generator["p1_build"]["margins_parquet_sha256"]

    entries = table.to_pandas()
    assert entries["entry_id"].str.startswith("synthetic-").all()
    assert entries["is_synthetic"].all()
    produced = (
        (entries["weight"] * entries["customs_value"])
        .groupby([entries["period"], entries["hts10"]])
        .sum()
    )
    assert produced[("2026-01", "8471300100")] == 12_345_678
    assert produced[("2026-02", "9903810100")] == 800
    assert "6109100012" not in set(entries["hts10"])
    assert str(entries.loc[0, "entry_date"])[:10].endswith("-15")


def test_two_builds_are_byte_identical(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    first = _run(margins_dir, tmp_path / "out-a")
    assert first.returncode == 0, first.stderr
    second = _run(margins_dir, tmp_path / "out-b")
    assert second.returncode == 0, second.stderr

    def hashes(out_dir: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out_dir.iterdir())
            if path.is_file()
        }

    hashes_a = hashes(tmp_path / "out-a")
    assert set(hashes_a) == {
        "synthetic_import_entries.parquet",
        "assumptions.json",
        "validation_report.json",
    }
    assert hashes_a == hashes(tmp_path / "out-b")


def test_substituted_margins_parquet_is_refused(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    tampered = pd.read_parquet(margins_dir / "margins_hts10_country_month.parquet")
    tampered.loc[1, "con_val_mo"] = 90_001
    tampered.to_parquet(
        margins_dir / "margins_hts10_country_month.parquet", index=False
    )
    result = _run(margins_dir, tmp_path / "out")
    assert result.returncode != 0
    assert "did not certify" in result.stderr
    assert not (tmp_path / "out" / "synthetic_import_entries.parquet").exists()


def test_substituted_cbp_page_is_refused(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    page_path = margins_dir / "cbp_newsroom_stats_trade.html"
    page_path.write_bytes(page_path.read_bytes().replace(b"83,133,856", b"93,133,856"))
    result = _run(margins_dir, tmp_path / "out")
    assert result.returncode != 0
    assert "did not certify" in result.stderr


def test_missing_p1_report_is_refused(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    (margins_dir / "build_report.json").unlink()
    result = _run(margins_dir, tmp_path / "out")
    assert result.returncode != 0
    assert "build report" in result.stderr


def _assert_contract_types(contract: dict, entries: pd.DataFrame) -> None:
    """Enforce every declared contract type, not just column presence.

    Fails closed on contract types this helper does not know, so a
    contract change can never silently drop enforcement.
    """
    for name, field in contract["inputs"].items():
        column = field["entries_column"]
        assert column in entries.columns, f"contract input {name} missing"
        series = entries[column]
        declared = field["type"]
        if declared == "positive integer dollars":
            assert pd.api.types.is_integer_dtype(series), (
                f"{name}: contract declares integer dollars, dtype is {series.dtype}"
            )
            assert (series > 0).all(), f"{name}: values must be positive"
        elif declared == "string":
            assert pd.api.types.is_string_dtype(series), (
                f"{name}: contract declares string, dtype is {series.dtype}"
            )
        elif declared == "bool":
            assert pd.api.types.is_bool_dtype(series), (
                f"{name}: contract declares bool, dtype is {series.dtype}"
            )
        elif declared == "date":
            parsed = pd.to_datetime(
                series.astype(str), format="%Y-%m-%d", errors="coerce"
            )
            assert parsed.notna().all(), (
                f"{name}: every value must parse as an ISO date"
            )
        else:
            raise AssertionError(
                f"contract declares unhandled type {declared!r} for {name}; "
                "extend _assert_contract_types before accepting it"
            )


def test_generated_entries_satisfy_the_pinned_engine_contract(tmp_path):
    contract = json.loads(CONTRACT.read_text())
    assert contract["source"]["commit"] == ("8b580e778a0ad134b93d563b5f8ae17980a29008")
    margins_dir = _margins_dir(tmp_path)
    result = _run(margins_dir, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    entries = pd.read_parquet(tmp_path / "out" / "synthetic_import_entries.parquet")
    _assert_contract_types(contract, entries)
    assert (entries["customs_value"] > 0).all()
    assert (entries["shipment_value"] == entries["customs_value"]).all()
    assert entries["hts_number"].str.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.\d{2}").all()
    assert entries["country_of_origin"].str.fullmatch(r"[A-Z]{2}").all()
    assert entries["is_postal_shipment"].dtype == bool
    assert not entries["is_postal_shipment"].any()
    engine_dates = entries.loc[entries["period"] >= "2026-02", "entry_date"]
    assert all(str(day)[:10] >= "2026-02-15" for day in engine_dates)


def test_contract_type_enforcement_rejects_floats_and_bad_dates(tmp_path):
    """The r2 probes: float monetary columns and an unparseable
    entry_date must fail the contract check, not pass on presence."""

    contract = json.loads(CONTRACT.read_text())
    margins_dir = _margins_dir(tmp_path)
    result = _run(margins_dir, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    entries = pd.read_parquet(tmp_path / "out" / "synthetic_import_entries.parquet")

    floated = entries.assign(customs_value=entries["customs_value"].astype("float64"))
    with pytest.raises(AssertionError, match="integer dollars"):
        _assert_contract_types(contract, floated)

    bad_dates = entries.copy()
    bad_dates["entry_date"] = bad_dates["entry_date"].astype(str)
    bad_dates.loc[bad_dates.index[0], "entry_date"] = "not-a-date"
    with pytest.raises(AssertionError, match="parse as an ISO date"):
        _assert_contract_types(contract, bad_dates)


@pytest.mark.skipif(not CONTRACT.is_file(), reason="contract fixture missing")
def test_contract_fixture_names_its_upstream_pin():
    contract = json.loads(CONTRACT.read_text())
    assert contract["source"]["repo"] == "TheAxiomFoundation/axiom-oracles"
    assert len(contract["source"]["file_sha256"]) == 64
    axiom_inputs = {
        field["axiom_input"]
        for field in contract["inputs"].values()
        if field["axiom_input"].startswith("us:policies")
    }
    assert axiom_inputs == {
        "us:policies/cbp/us-tariff-duty/composition#input.customs_value",
        "us:policies/cbp/us-tariff-duty/composition#input.shipment_value",
        "us:policies/cbp/us-tariff-duty/composition#input.hts_number",
        "us:policies/cbp/us-tariff-duty/composition#input.country_of_origin",
        "us:policies/cbp/us-tariff-duty/composition#input.is_postal_shipment",
    }


def test_cli_fails_when_margins_missing_from_window(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    result = _run(margins_dir, tmp_path / "entries", start="2027-01")
    assert result.returncode == 1
    assert "no margin cells" in result.stdout


def test_override_build_registers_override_basis(tmp_path):
    """The r2 probe: a supported override build must emit a register that
    describes its overrides — never the CBP-anchored claims — while still
    carrying the P1 authentication block."""

    margins_dir = _margins_dir(tmp_path)
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--margins-dir",
            str(margins_dir),
            "--out-dir",
            str(out_dir),
            "--start",
            "2026-01",
            "--mean-entry-value",
            "1000",
            "--informal-share",
            "0.5",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    register = json.loads((out_dir / "assumptions.json").read_text())
    anchors = register["size_model"]["anchor_provenance"]
    assert anchors["basis"] == "explicit overrides"
    assert anchors["mean_source"] == "override"
    assert anchors["share_source"] == "override"
    assert "CBP's published" not in register["statement"]
    assert "explicitly overridden mean entry value" in register["statement"]
    assert "explicitly overridden informal count share" in register["statement"]
    assert register["p1_build"]["margins_parquet_sha256"]


# --- Cross-cell aggregation exactness (sol r3 blocking finding) -----------
#
# The CLI's own aggregations — the chapter x country cross-check and the
# monthly weighted-entry counts — accumulate across cells, where totals are
# not int64-bounded even though every cell is. These probes pin the exact
# overflow frames from the review: int64 accumulation wrapped both sides of
# the cross-check to -2**63 (passing modulo 2**64) and emitted -2**63
# monthly counts; Python-int accumulation must yield the true positive
# totals and refuse wrapped equalities.


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("build_us_import_entries_probe", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chapter_cross_check_is_exact_beyond_int64():
    """1,024 margin cells of 2**53 in one chapter: the chapter total is the
    true +2**63, not an identical -2**63 wrap on both sides."""

    cli = _load_cli_module()
    n = 1024
    margins = pd.DataFrame(
        {
            "period": ["2026-03"] * n,
            "hts10": [f"01{i:08d}" for i in range(n)],
            "cty_code": ["5700"] * n,
            "con_val_mo": [2**53] * n,
        }
    )
    entries = pd.DataFrame(
        {
            "period": ["2026-03"] * n,
            "chapter": ["01"] * n,
            "census_country_code": ["5700"] * n,
            "weight": [1] * n,
            "customs_value": [2**53] * n,
        }
    )
    assert cli._chapter_country_cross_check(entries, margins) == {
        "chapter_country_cells_exact": 1
    }
    totals = cli._exact_group_sums(
        entries["weight"].astype(object) * entries["customs_value"].astype(object),
        [
            entries["period"].astype(str),
            entries["chapter"].astype(str),
            entries["census_country_code"].astype(str),
        ],
    )
    assert totals.to_list() == [2**63]
    assert totals.iloc[0] > 0


def test_chapter_cross_check_refuses_wrapped_equality():
    """A mod-2**64 collision must fail: entries whose true chapter total is
    2**64 + 10 wrapped to the margin total 10 under int64 accumulation, so
    the pre-fix comparison passed on identical wraps."""

    cli = _load_cli_module()
    margins = pd.DataFrame(
        {
            "period": ["2026-03"],
            "hts10": ["0101210001"],
            "cty_code": ["5700"],
            "con_val_mo": [10],
        }
    )
    entries = pd.DataFrame(
        {
            "period": ["2026-03"] * 5,
            "chapter": ["01"] * 5,
            "census_country_code": ["5700"] * 5,
            "weight": [1] * 5,
            "customs_value": [2**62] * 4 + [10],
        }
    )
    with pytest.raises(ValueError, match="does not match margins"):
        cli._chapter_country_cross_check(entries, margins)


def test_monthly_counts_survive_totals_beyond_int64():
    """2,048 cells at mean entry value 2 carry 2**52 weighted entries each;
    the monthly total is the true +2**63, where int64 wrapped to -2**63."""

    cli = _load_cli_module()
    n = 2048
    entries = pd.DataFrame(
        {
            "period": ["2026-03"] * n,
            "weight": [2**52] * n,
            "customs_value": [2] * n,
        }
    )
    counts = cli._monthly_counts(entries)
    assert counts == {"2026-03": 2**63}
    assert counts["2026-03"] > 0
