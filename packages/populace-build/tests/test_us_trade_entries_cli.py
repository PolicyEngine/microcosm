"""End-to-end test for the synthetic import-entry build CLI (#615 P2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

GOLDEN = Path(__file__).parent / "golden" / "us_trade"
CLI = Path(__file__).resolve().parents[3] / "tools" / "build_us_import_entries.py"


def _margins_dir(tmp_path: Path) -> Path:
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
    frame.to_parquet(margins_dir / "margins_hts10_country_month.parquet", index=False)
    cbp = (GOLDEN / "cbp_trade_stats_fragment.html").read_bytes()
    (margins_dir / "cbp_newsroom_stats_trade.html").write_bytes(cbp)
    return margins_dir


def test_cli_builds_labeled_validated_entries(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    out_dir = tmp_path / "entries"
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["exact"] is True
    assert report["cells_checked"] == 4  # 2025-12 outside window, zero cell skipped
    assert report["window"] == {"start": "2026-01", "end": "2026-02", "months": 2}
    assert report["engine_runnable_months"] == ["2026-02"]
    assert report["chapter_country_cells_exact"] == 4

    register = json.loads((out_dir / "assumptions.json").read_text())
    assert register["synthetic"] is True
    anchors = register["size_model"]["anchor_provenance"]
    assert anchors["fiscal_year"] == 2026
    assert anchors["total_entry_summaries"] == 83_133_856
    assert 0.68 < anchors["informal_count_share"] < 0.69
    assert 30_000 < anchors["mean_entry_value"] < 36_000

    table = pq.read_table(out_dir / "synthetic_import_entries.parquet")
    metadata = table.schema.metadata
    assert metadata[b"populace_us_trade.synthetic"] == b"true"
    generator = json.loads(metadata[b"populace_us_trade.generator"])
    assert generator["issue"] == "PolicyEngine/populace#615"
    assert generator["synthetic"] is True

    entries = table.to_pandas()
    assert entries["entry_id"].str.startswith("synthetic-").all()
    produced = (
        (entries["weight"] * entries["customs_value"])
        .groupby([entries["period"], entries["hts10"]])
        .sum()
    )
    assert produced[("2026-01", "8471300100")] == 12_345_678
    assert produced[("2026-02", "9903810100")] == 800
    assert "6109100012" not in set(entries["hts10"])
    assert str(entries.loc[0, "entry_date"])[:10].endswith("-15")


def test_cli_fails_when_margins_missing_from_window(tmp_path):
    margins_dir = _margins_dir(tmp_path)
    out_dir = tmp_path / "entries"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--margins-dir",
            str(margins_dir),
            "--out-dir",
            str(out_dir),
            "--start",
            "2027-01",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "no margin cells" in result.stdout
