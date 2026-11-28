"""The country-neutral comparison of model totals with official receipts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from microcosm.build.firms.receipts import (
    INPUT_NOT_POPULATED,
    NO_MODEL_TOTAL,
    NO_OFFICIAL_FIGURE,
    RECEIPT_SOURCE_COLUMNS,
    Receipt,
    build_comparison_rows,
    compare_total,
    read_receipts,
    render_comparison_markdown,
)

_COLUMNS = ("levy", "value_eur", "period", *RECEIPT_SOURCE_COLUMNS, "note")
_SOURCE = {
    "measure": "cash receipts",
    "source_publisher": "Tax Office",
    "source_title": "Receipts bulletin",
    "edition": "March 2026",
    "url": "https://example.org/receipts",
    "accessed": "2026-11-20",
}


def _write(path: Path, rows: list[dict[str, str]], columns=_COLUMNS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _row(name: str, value: str, **overrides: str) -> dict[str, str]:
    return {
        "levy": name,
        "value_eur": value,
        "period": "2025",
        **_SOURCE,
        "note": "",
        **overrides,
    }


def _euros(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}€{abs(value) / 1e6:,.1f}m"


def _receipts(tmp_path: Path) -> dict[str, Receipt]:
    path = _write(
        tmp_path / "receipts.csv",
        [
            _row("payroll_levy", "200000000"),
            _row(
                "site_levy", "", note="Not published yet.", **dict.fromkeys(_SOURCE, "")
            ),
            _row("profit_levy", "50000000"),
        ],
    )
    return read_receipts(path, name_column="levy", value_column="value_eur")


def test_reader_keeps_order_and_blank_figures(tmp_path) -> None:
    receipts = _receipts(tmp_path)
    assert list(receipts) == ["payroll_levy", "site_levy", "profit_levy"]
    assert receipts["payroll_levy"].official_value == 200_000_000.0
    assert receipts["payroll_levy"].edition == "March 2026"
    assert receipts["site_levy"].official_value is None


def test_reader_refuses_missing_columns_repeats_and_unsourced_figures(
    tmp_path,
) -> None:
    short = _write(tmp_path / "a.csv", [], columns=_COLUMNS[:-1])
    with pytest.raises(ValueError, match=r"missing columns \['note'\]"):
        read_receipts(short, name_column="levy", value_column="value_eur")
    repeated = _write(tmp_path / "b.csv", [_row("x", "1"), _row("x", "2")])
    with pytest.raises(ValueError, match="duplicate levy 'x'"):
        read_receipts(repeated, name_column="levy", value_column="value_eur")
    unsourced = _write(tmp_path / "c.csv", [_row("x", "1", url="")])
    with pytest.raises(ValueError, match=r"x has a figure but no \['url'\]"):
        read_receipts(unsourced, name_column="levy", value_column="value_eur")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"model_total": 150e6, "populated": True, "official_value": 200e6},
            ("€150.0m", "€200.0m", "-€50.0m (-25.0%)", "0.750"),
        ),
        (
            {"model_total": 0.0, "populated": False, "official_value": 200e6},
            (INPUT_NOT_POPULATED, "€200.0m", INPUT_NOT_POPULATED, INPUT_NOT_POPULATED),
        ),
        (
            {"model_total": None, "populated": True, "official_value": 200e6},
            (NO_MODEL_TOTAL, "€200.0m", NO_MODEL_TOTAL, NO_MODEL_TOTAL),
        ),
        (
            {"model_total": 5e6, "populated": True, "official_value": None},
            ("€5.0m", NO_OFFICIAL_FIGURE, NO_OFFICIAL_FIGURE, NO_OFFICIAL_FIGURE),
        ),
        (
            {"model_total": 5e6, "populated": True, "official_value": 0.0},
            ("€5.0m", "€0.0m", "€5.0m", "official figure is zero"),
        ),
    ],
)
def test_compare_total_never_invents_a_gap(kwargs, expected) -> None:
    row = compare_total("levy", reasons=["One.", "Two."], format_value=_euros, **kwargs)
    assert (row.model, row.official, row.gap, row.ratio) == expected
    assert row.reason == "One. Two."


def test_rows_follow_receipts_then_model_only_names(tmp_path) -> None:
    receipts = _receipts(tmp_path)
    seen = []

    def reasons_for(name, receipt):
        seen.append((name, receipt is not None))
        return [f"Official measure: {receipt.measure}."] if receipt else []

    rows = build_comparison_rows(
        {"payroll_levy": 180e6, "site_levy": 0.0, "extra_credit": 3e6},
        {"payroll_levy": True, "site_levy": False, "extra_credit": True},
        receipts,
        reasons_for=reasons_for,
        format_value=_euros,
    )
    assert [r.name for r in rows] == [
        "payroll_levy",
        "site_levy",
        "profit_levy",
        "extra_credit",
    ]
    assert rows[1].gap == INPUT_NOT_POPULATED
    assert rows[2].model == NO_MODEL_TOTAL
    assert rows[3].official == NO_OFFICIAL_FIGURE
    assert rows[0].reason == "Official measure: cash receipts."
    assert seen[-1] == ("extra_credit", False)


def test_rows_refuse_a_total_without_a_populated_flag(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\['payroll_levy'\] have no true/false"):
        build_comparison_rows(
            {"payroll_levy": 1.0},
            {"payroll_levy": "yes"},
            _receipts(tmp_path),
            reasons_for=lambda name, receipt: [],
            format_value=_euros,
        )


def test_markdown_escapes_cells_and_lists_sources(tmp_path) -> None:
    receipts = _receipts(tmp_path)
    rows = build_comparison_rows(
        {"payroll_levy": 180e6},
        {"payroll_levy": True},
        receipts,
        reasons_for=lambda name, receipt: ["a | b"],
        format_value=_euros,
    )
    report = render_comparison_markdown(
        rows, receipts, title="Firm levies, 2025", name_header="Levy", note="run 7"
    )
    lines = report.splitlines()
    assert lines[0] == "# Firm levies, 2025"
    assert "Model run note: run 7" in lines
    assert "| Levy | Model total | Official | Gap | Ratio | Reason / caveat |" in lines
    assert (
        "| payroll_levy | €180.0m | €200.0m | -€20.0m (-10.0%) | 0.900 | a \\| b |"
        in lines
    )
    assert "- `site_levy`: no official figure fetched. Not published yet." in lines
    assert (
        "- `payroll_levy`: Tax Office, Receipts bulletin; March 2026; "
        "https://example.org/receipts (accessed 2026-11-20). "
    ) in lines
    assert report.endswith("\n")
