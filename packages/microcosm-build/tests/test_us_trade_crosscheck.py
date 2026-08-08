"""Offline tests for the API cross-check comparator (microcosm#615 P1)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "crosscheck_us_import_margins_api",
    REPO_ROOT / "tools" / "crosscheck_us_import_margins_api.py",
)
crosscheck = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(crosscheck)

_HEADER = [
    "CTY_CODE",
    "SUMMARY_LVL",
    "CON_VAL_MO",
    "GEN_VAL_MO",
    "CAL_DUT_MO",
    "DUT_VAL_MO",
    "CON_QY1_MO",
    "GEN_QY1_MO",
    "UNIT_QY1",
    "I_COMMODITY",
    "time",
]


def _api_payload(con_val: str = "1000") -> bytes:
    rows = [
        _HEADER,
        [
            "1220",
            "DET",
            con_val,
            "1000",
            "25",
            "400",
            "3",
            "3",
            "NO",
            "0101210010",
            "2026-01",
        ],
        [
            "-",
            "DET",
            con_val,
            "1000",
            "25",
            "400",
            "3",
            "3",
            "NO",
            "0101210010",
            "2026-01",
        ],
    ]
    return json.dumps(rows).encode("utf-8")


def _margins() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "2026-01",
                "hts10": "0101210010",
                "chapter": "01",
                "cty_code": "1220",
                "iso2": "CA",
                "con_val_mo": 1000,
                "gen_val_mo": 1000,
                "cal_dut_mo": 25,
                "dut_val_mo": 400,
                "con_qy1_mo": 3,
                "gen_qy1_mo": 3,
            }
        ]
    )


def _totals(con_val: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": "2026-01",
                "hts10": "0101210010",
                "con_val_mo": con_val,
                "gen_val_mo": 1000,
                "cal_dut_mo": 25,
                "dut_val_mo": 400,
            },
            # A YTD carrier: all-zero control row the API never publishes.
            {
                "period": "2026-01",
                "hts10": "0101999999",
                "con_val_mo": 0,
                "gen_val_mo": 0,
                "cal_dut_mo": 0,
                "dut_val_mo": 0,
            },
        ]
    )


def test_agreeing_pair_reports_zero_failures(tmp_path):
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, _api_payload()),
    )
    assert report["cells_compared"] == 1
    assert report["dollar_mismatch_cells"] == 0
    assert report["total_mismatches"] == 0
    assert report["api_reconciliation_failures"] == 0
    assert report["api_totals_absent"] == 0
    assert report["api_only_cells"] == 0
    assert report["bulk_only_cells"] == 0


def test_value_disagreement_and_total_drift_are_counted(tmp_path):
    # Bulk margins carry 1000; the API answers 1001 → cell mismatch, and
    # the API's own detail-vs-totals reconciliation stays internally
    # consistent so the divergence is attributed to the channels.
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(con_val=999),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, _api_payload(con_val="1001")),
    )
    assert report["dollar_mismatch_cells"] == 1
    assert report["total_mismatches"] == 1
    assert any(
        detail["measure"] == "con_val_mo" and detail["api"] == 1001
        for detail in report["mismatches"]
    )


def test_zero_carrier_totals_do_not_count_as_missing(tmp_path):
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, _api_payload()),
    )
    # The all-zero 0101999999 control row must not appear as a mismatch.
    assert report["total_mismatches"] == 0


def test_parse_sample_round_trip():
    sample = crosscheck._parse_sample(["2025-01:85,87", "2026-05:01"])
    assert sample == (("2025-01", ("85", "87")), ("2026-05", ("01",)))


def test_default_sample_covers_both_window_endpoints():
    months = {month for month, _ in crosscheck.DEFAULT_SAMPLE}
    assert "2025-01" in months
    assert "2026-06" in months


def test_absent_api_totals_is_counted_and_retrievals_are_recorded(tmp_path):
    # An API response with detail rows but no publisher '-' totals row:
    # the totals leg has no counterparty, which must be counted (it gates
    # in main), and the pair must carry its retrieval manifest.
    rows = [_HEADER] + [_api_payload_row("1220")]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["api_totals_absent"] == 1
    assert report["api_retrievals"]
    assert all(entry.get("sha256") for entry in report["api_retrievals"])
    # The API pull carries active detail with no '-' total to reconcile
    # against, which its own internal reconciliation now counts as a
    # coverage failure; both signals gate in main's sum.
    assert report["api_reconciliation_failures"] == 1


def _api_payload_row(cty: str) -> list[str]:
    return [
        cty,
        "DET",
        "1000",
        "1000",
        "25",
        "400",
        "3",
        "3",
        "NO",
        "0101210010",
        "2026-01",
    ]


def test_api_ytd_union_zero_rows_are_agreement_by_absence(tmp_path):
    # After January the API returns YTD-union rows whose every month
    # measure is zero; the bulk detail omits pairs with no activity that
    # month. Zero on one channel and silence on the other is agreement —
    # counted, not gated.
    rows = [
        _HEADER,
        _api_payload_row("1220"),
        ["1330", "DET", "0", "0", "0", "0", "0", "0", "NO", "0101210010", "2026-01"],
        [
            "-",
            "DET",
            "1000",
            "1000",
            "25",
            "400",
            "3",
            "3",
            "NO",
            "0101210010",
            "2026-01",
        ],
    ]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["cells_compared"] == 1
    assert report["dollar_mismatch_cells"] == 0
    assert report["total_mismatches"] == 0
    assert report["api_reconciliation_failures"] == 0
    assert report["api_only_cells"] == 0
    assert report["api_zero_union_cells"] == 1
    assert report["bulk_zero_union_cells"] == 0


def test_api_zero_union_totals_are_agreement_by_absence(tmp_path):
    # The API's '-' rows echo the same YTD zero carriers the bulk control
    # file holds. Zero-vs-zero (and zero-vs-absent) on the totals leg is
    # agreement — counted per side, never gated.
    rows = [
        _HEADER,
        _api_payload_row("1220"),
        [
            "-",
            "DET",
            "1000",
            "1000",
            "25",
            "400",
            "3",
            "3",
            "NO",
            "0101210010",
            "2026-01",
        ],
        ["-", "DET", "0", "0", "0", "0", "0", "0", "NO", "0101999999", "2026-01"],
    ]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["total_mismatches"] == 0
    assert report["api_reconciliation_failures"] == 0
    assert report["api_zero_union_totals"] == 1
    assert report["bulk_zero_union_totals"] == 1


def test_api_only_nonzero_total_still_gates(tmp_path):
    # A commodity whose API total carries value but which the bulk totals
    # lack entirely is genuine one-sided divergence on both legs.
    rows = [
        _HEADER,
        _api_payload_row("1220"),
        [
            "1330",
            "DET",
            "700",
            "700",
            "7",
            "70",
            "1",
            "1",
            "NO",
            "0102000000",
            "2026-01",
        ],
        [
            "-",
            "DET",
            "1000",
            "1000",
            "25",
            "400",
            "3",
            "3",
            "NO",
            "0101210010",
            "2026-01",
        ],
        ["-", "DET", "700", "700", "7", "70", "1", "1", "NO", "0102000000", "2026-01"],
    ]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["total_mismatches"] == 1
    assert report["dollar_mismatch_cells"] == 1
    assert report["api_only_cells"] == 1
    assert report["api_zero_union_totals"] == 0


def test_api_only_active_cell_still_gates(tmp_path):
    # A one-sided API cell carrying a nonzero measure is genuine
    # divergence — the bulk channel is missing published activity — and
    # must keep gating on both the cell and totals legs.
    rows = [
        _HEADER,
        _api_payload_row("1220"),
        [
            "1330",
            "DET",
            "500",
            "500",
            "10",
            "200",
            "1",
            "1",
            "NO",
            "0101210010",
            "2026-01",
        ],
        [
            "-",
            "DET",
            "1500",
            "1500",
            "35",
            "600",
            "4",
            "4",
            "NO",
            "0101210010",
            "2026-01",
        ],
    ]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals(),
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["dollar_mismatch_cells"] == 1
    assert report["total_mismatches"] == 4  # 4 dollar measures diverge on the total
    assert report["api_only_cells"] == 1
    assert report["api_zero_union_cells"] == 0
    assert any(detail["measure"] == "api_only" for detail in report["mismatches"])


def test_main_gates_on_zero_cells_and_absent_totals(tmp_path, monkeypatch):
    margins_path = tmp_path / "margins.parquet"
    totals_path = tmp_path / "totals.parquet"
    _margins().to_parquet(margins_path, index=False)
    _totals().to_parquet(totals_path, index=False)
    out_path = tmp_path / "report.json"

    def fake_pair(month, chapter, margins, totals, api_key, cache_dir, **kwargs):
        return {
            "month": month,
            "chapter": chapter,
            "cells_compared": 0,
            "dollar_mismatch_cells": 0,
            "total_mismatches": 0,
            "api_reconciliation_failures": 0,
            "api_totals_absent": 0,
            "api_only_cells": 0,
            "bulk_only_cells": 0,
            "api_zero_union_cells": 0,
            "bulk_zero_union_cells": 0,
            "api_zero_union_totals": 0,
            "bulk_zero_union_totals": 0,
            "quantity_diff_cells": 0,
            "api_null_quantity_cells": 0,
            "api_retrievals": [],
            "mismatches": [],
        }

    monkeypatch.setattr(crosscheck, "_compare_pair", fake_pair)
    monkeypatch.setattr(
        "sys.argv",
        [
            "crosscheck",
            "--margins",
            str(margins_path),
            "--totals",
            str(totals_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(out_path),
            "--census-key",
            "test-key",
            "--pair",
            "2026-01:01",
        ],
    )
    # Zero compared cells across the whole run: an empty comparison must
    # exit nonzero even with zero recorded mismatches.
    assert crosscheck.main() == 1
    report = json.loads(out_path.read_text())
    assert report["empty_evidence"] is True
    assert report["gating_failures"] == 0
    assert report["inputs"]["margins_parquet_sha256"]
    assert report["inputs"]["census_totals_parquet_sha256"]

    def fake_pair_absent(month, chapter, *args, **kwargs):
        pair = fake_pair(month, chapter, *args, **kwargs)
        return {**pair, "cells_compared": 5, "api_totals_absent": 1}

    monkeypatch.setattr(crosscheck, "_compare_pair", fake_pair_absent)
    # A missing totals counterparty gates even when cells were compared.
    assert crosscheck.main() == 1
    report = json.loads(out_path.read_text())
    assert report["empty_evidence"] is False
    assert report["api_totals_absent_pairs"] == 1
    assert report["gating_failures"] == 1


def test_zero_dollar_quantity_bearing_total_is_active_not_zero_union(tmp_path):
    # The r2 probe: a bulk total with zero dollars and quantity 7 is
    # published activity under the dollars-or-published-quantities
    # predicate. With no API totals slice at all it must surface as an
    # absent totals counterparty — never be reclassified as a YTD zero
    # carrier with every gate at zero.
    rows = [_HEADER, _api_payload_row("1220")]
    payload = json.dumps(rows).encode("utf-8")
    totals = pd.DataFrame(
        [
            {
                "period": "2026-01",
                "hts10": "0101999999",
                "con_val_mo": 0,
                "gen_val_mo": 0,
                "cal_dut_mo": 0,
                "dut_val_mo": 0,
                "con_qy1_mo": 7,
                "gen_qy1_mo": 0,
            }
        ]
    )
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        totals,
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["bulk_zero_union_totals"] == 0
    assert report["api_totals_absent"] == 1


def test_zero_dollar_quantity_bearing_total_gates_one_sided(tmp_path):
    # The same quantity-bearing total against a present API totals slice
    # that lacks the commodity is one-sided divergence on the totals leg.
    rows = [
        _HEADER,
        _api_payload_row("1220"),
        [
            "-",
            "DET",
            "1000",
            "1000",
            "25",
            "400",
            "3",
            "3",
            "NO",
            "0101210010",
            "2026-01",
        ],
    ]
    payload = json.dumps(rows).encode("utf-8")
    totals = _totals()
    totals.loc[totals["hts10"] == "0101999999", "con_qy1_mo"] = 7
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        totals,
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["bulk_zero_union_totals"] == 0
    assert report["total_mismatches"] == 1


def test_active_pair_with_no_totals_evidence_on_either_side_gates(tmp_path):
    # The r2 probe: matching dollar-active detail with BOTH totals slices
    # empty compared one cell and passed. An active pair whose totals leg
    # compared nothing has absent evidence — the API totals absence must
    # gate even when the bulk totals slice is empty too.
    rows = [_HEADER, _api_payload_row("1220")]
    payload = json.dumps(rows).encode("utf-8")
    other_chapter_totals = _totals()
    other_chapter_totals["hts10"] = "0201000000"
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        other_chapter_totals,
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["cells_compared"] == 1
    assert report["dollar_mismatch_cells"] == 0
    assert report["api_totals_absent"] == 1


def test_inactive_pair_with_no_totals_anywhere_does_not_false_gate(tmp_path):
    # A genuinely empty pair — the API answers 204 with no rows at all,
    # and both bulk slices are empty. The r2 fix's early empty-API branch
    # hard-coded api_totals_absent=1 here, false-gating a pair with
    # nothing to certify on either channel; every pair now flows through
    # the one comparison path, where no activity and no active totals
    # anywhere means no absent-evidence gate (the run-level
    # empty-evidence gate covers all-empty runs).
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins().iloc[:0],
        _totals().iloc[:0],
        "test-key",
        tmp_path,
        fetch=lambda url: (204, b""),
    )
    assert report["api_totals_absent"] == 0
    assert report["cells_compared"] == 0
    assert report["dollar_mismatch_cells"] == 0
    assert report["total_mismatches"] == 0
    assert report["api_reconciliation_failures"] == 0


def test_inactive_pair_with_zero_carrier_detail_does_not_false_gate(tmp_path):
    # The same pair expressed through the API's YTD union: one all-zero
    # detail row instead of silence. Zero carriers are inactivity under
    # the shared predicate, so the verdict must match the truly-empty
    # wire shape above.
    rows = [
        _HEADER,
        ["1330", "DET", "0", "0", "0", "0", "0", "0", "NO", "0101210010", "2026-01"],
    ]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins().iloc[:0],
        _totals().iloc[:0],
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["api_totals_absent"] == 0
    assert report["cells_compared"] == 0


def test_zero_carrier_api_totals_are_not_totals_evidence_for_an_active_pair(
    tmp_path,
):
    # The r3 probe: matching active detail, an empty bulk totals slice,
    # and an API totals slice holding exactly one unrelated all-zero
    # carrier. Absence was tested on the raw slice, so the zero carrier
    # stood in as "totals evidence" while the active-row join compared
    # nothing — one compared cell, api_totals_absent=0, zero
    # reconciliation failures. Absence is now judged after the activity
    # predicate, and the API's internal reconciliation counts the active
    # detail left without any published total.
    rows = [
        _HEADER,
        _api_payload_row("1220"),
        ["-", "DET", "0", "0", "0", "0", "0", "0", "NO", "0101999999", "2026-01"],
    ]
    payload = json.dumps(rows).encode("utf-8")
    report = crosscheck._compare_pair(
        "2026-01",
        "01",
        _margins(),
        _totals().iloc[:0],
        "test-key",
        tmp_path,
        fetch=lambda url: (200, payload),
    )
    assert report["cells_compared"] == 1
    assert report["api_totals_absent"] == 1
    assert report["api_zero_union_totals"] == 1
    assert report["api_reconciliation_failures"] == 1
