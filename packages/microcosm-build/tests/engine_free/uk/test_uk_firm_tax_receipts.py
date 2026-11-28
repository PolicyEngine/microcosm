"""The UK firm tax receipts comparison tool and its official-figure fixture."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from functools import cache

import pytest

from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
TOOL = PATHS.repository / "tools" / "compare_uk_firm_taxes_to_receipts.py"
RECEIPTS = PATHS.tests / "fixtures" / "uk" / "firm_tax_receipts_2024_25.csv"
TAXES = (
    "firm_vat",
    "firm_employer_ni",
    "apprenticeship_levy",
    "corporation_tax",
    "bank_surcharge",
    "firm_business_rates",
)


@cache
def _load_tool():
    name = "compare_uk_firm_taxes_to_receipts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _receipt(tool, tax: str, value: float | None):
    return tool.Receipt(
        name=tax,
        official_value=value,
        period="2024-25",
        measure="test measure",
        source_publisher="p",
        source_title="t",
        edition="e",
        url="https://example.org",
        accessed="2026-11-22",
        note="",
    )


def test_fixture_has_required_columns_and_sourced_figures() -> None:
    tool = _load_tool()
    with RECEIPTS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert set(tool.REQUIRED_COLUMNS) <= set(reader.fieldnames or ())
        rows = list(reader)
    assert [row["tax"] for row in rows] == list(TAXES)
    for row in rows:
        assert row["period"] == "2024-25"
        if row["official_value_gbp"].strip():
            assert float(row["official_value_gbp"]) > 0
            for column in ("url", "edition", "accessed", "measure", "source_title"):
                assert row[column].strip(), (row["tax"], column)
            assert row["url"].startswith("https://")
        else:
            # A blank figure must say why it was not fetched.
            assert row["note"].strip(), row["tax"]
    # The reader enforces the same rule.
    assert set(tool.read_receipts(RECEIPTS)) == set(TAXES)


def test_every_provenance_source_has_a_content_hash() -> None:
    provenance = json.loads(
        RECEIPTS.with_name(RECEIPTS.name + ".provenance.json").read_text(
            encoding="utf-8"
        )
    )
    sources = provenance["sources"]
    assert sources
    covered = set()
    for name, source in sources.items():
        digest = source.get("sha256", "")
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name
        assert source["url"].startswith("https://assets.publishing.service.gov.uk/")
        covered.update(source["taxes"])
    assert covered == set(TAXES)


def test_reader_refuses_a_figure_without_a_source(tmp_path) -> None:
    tool = _load_tool()
    path = tmp_path / "receipts.csv"
    header = ",".join(tool.REQUIRED_COLUMNS)
    path.write_text(f"{header}\nfirm_vat,1,2024-25,m,p,t,,https://x,2026-11-22,\n")
    with pytest.raises(ValueError, match="edition"):
        tool.read_receipts(path)


def test_gap_arithmetic() -> None:
    tool = _load_tool()
    row = tool.compare_tax(
        "corporation_tax",
        model_total=80e9,
        populated=True,
        receipt=_receipt(tool, "corporation_tax", 100e9),
    )
    assert row.model == "£80.00bn"
    assert row.official == "£100.00bn"
    assert row.gap == "-£20.00bn (-20.0%)"
    assert row.ratio == "0.800"


def test_blank_official_figure_prints_message() -> None:
    tool = _load_tool()
    row = tool.compare_tax(
        "firm_employer_ni",
        model_total=50e9,
        populated=True,
        receipt=_receipt(tool, "firm_employer_ni", None),
    )
    assert row.official == tool.NO_OFFICIAL_FIGURE == "no official figure fetched"
    assert row.gap == row.ratio == "no official figure fetched"
    assert "%" not in row.gap


def test_unpopulated_input_prints_no_gap_or_percentage() -> None:
    tool = _load_tool()
    row = tool.compare_tax(
        "firm_business_rates",
        model_total=0.0,
        populated=False,
        receipt=_receipt(tool, "firm_business_rates", 26e9),
    )
    assert row.model == row.gap == row.ratio == "input not populated"
    assert all("%" not in cell for cell in (row.model, row.gap, row.ratio, row.reason))


def test_report_end_to_end(tmp_path) -> None:
    tool = _load_tool()
    taxes = {
        "period": 2024,
        "totals": {tax: 1e9 for tax in TAXES},
        "inputs_populated": {tax: tax == "firm_vat" for tax in TAXES},
        "note": "synthetic",
    }
    path = tmp_path / "taxes.json"
    path.write_text(json.dumps(taxes))
    out = tmp_path / "report.md"
    assert tool.main(["--taxes-json", str(path), "--out", str(out)]) == 0
    report = out.read_text(encoding="utf-8")
    lines = {line.split(" | ")[0].strip("| "): line for line in report.splitlines()}
    assert "in-sample" in lines["firm_vat"].lower()
    assert "does not validate the model" in lines["firm_vat"]
    for tax in TAXES[1:]:
        assert "input not populated" in lines[tax]
        assert "%" not in lines[tax]
    assert "England only" in lines["firm_business_rates"]
    for tax in ("firm_employer_ni", "apprenticeship_levy"):
        assert "not a validation" in lines[tax]
        assert "does not set it" in lines[tax]
        assert "public sector" not in lines[tax]
    assert "firm_paybill_below_secondary_threshold" in lines["firm_employer_ni"]
    assert "firm_is_single_director_company" in lines["firm_employer_ni"]
    assert "cash" in lines["firm_vat"]
    # Only the populated VAT row carries a percentage gap.
    assert "%" in lines["firm_vat"]


def test_tax_missing_from_the_model_json_is_not_called_unpopulated() -> None:
    tool = _load_tool()
    receipts = tool.read_receipts(RECEIPTS)
    rows = {
        row.name: row
        for row in tool.build_rows({"period": 2024, "totals": {}}, receipts)
    }
    assert rows["bank_surcharge"].model == "no model total supplied"
    assert "%" not in rows["bank_surcharge"].gap


def test_paybill_caveat_survives_without_receipt_or_caveat_entry() -> None:
    tool = _load_tool()
    for tax in tool.PAYBILL_TAXES:
        row = tool.compare_tax(tax, model_total=1e9, populated=True, receipt=None)
        assert row.reason.startswith(tool.PAYBILL_ASSUMED_CAVEAT)


@pytest.mark.parametrize(
    "inputs_populated", [None, {}, {"firm_employer_ni": "yes"}, {"firm_vat": True}]
)
def test_total_without_explicit_populated_flag_is_refused(inputs_populated) -> None:
    tool = _load_tool()
    receipts = tool.read_receipts(RECEIPTS)
    taxes = {"period": 2024, "totals": {"firm_employer_ni": 0.0}}
    if inputs_populated is not None:
        taxes["inputs_populated"] = inputs_populated
    with pytest.raises(ValueError, match="firm_employer_ni"):
        tool.compare(taxes, receipts)


def test_period_mismatch_is_refused() -> None:
    tool = _load_tool()
    receipts = tool.read_receipts(RECEIPTS)
    with pytest.raises(ValueError, match="2025-26"):
        tool.compare({"period": 2025, "totals": {}}, receipts)
