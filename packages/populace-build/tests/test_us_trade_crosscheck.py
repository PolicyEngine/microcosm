"""Offline tests for the API cross-check comparator (populace#615 P1)."""

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
