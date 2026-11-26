#!/usr/bin/env python3
"""Compare weighted UK firm tax totals with official 2024-25 receipts.

Another process runs policyengine-uk on the generated firms and writes the
weighted totals as JSON::

    {
      "period": 2024,
      "totals": {"firm_vat": 1.7e11, ...},          # GBP, summed with firm_weight
      "inputs_populated": {"firm_vat": true, ...},  # false: the tax's inputs are unset
      "note": "free text"
    }

This tool sets those totals beside the official figures in
``tests/fixtures/uk/firm_tax_receipts_2024_25.csv`` (HMRC cash receipts and
MHCLG NNDR3, each with its URL, edition and access date) and prints a markdown
table: model total, official figure, gap (model less official, and ratio) and
the reasons the two may not measure the same thing.

Usage::

    uv run --no-sync python tools/compare_uk_firm_taxes_to_receipts.py \
        --taxes-json firm_tax_totals.json [--out comparison.md]

Every tax in ``totals`` needs an explicit true/false ``inputs_populated`` flag;
the tool refuses the JSON otherwise. A tax whose inputs are not populated prints
"input not populated" in place of a gap, never a -100% shortfall. A tax with no official figure prints "no official
figure fetched". The comparison is a diagnostic, not a calibration target.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.firms.receipts import (
    INPUT_NOT_POPULATED as INPUT_NOT_POPULATED,
)
from microcosm.build.firms.receipts import (
    NO_MODEL_TOTAL as NO_MODEL_TOTAL,
)
from microcosm.build.firms.receipts import (
    NO_OFFICIAL_FIGURE as NO_OFFICIAL_FIGURE,
)
from microcosm.build.firms.receipts import (
    RECEIPT_SOURCE_COLUMNS,
    ComparisonRow,
    Receipt,
    build_comparison_rows,
    compare_total,
    render_comparison_markdown,
)
from microcosm.build.firms.receipts import read_receipts as _read_receipts

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "tests"
    / "fixtures"
    / "uk"
    / "firm_tax_receipts_2024_25.csv"
)

REQUIRED_COLUMNS = (
    "tax",
    "official_value_gbp",
    "period",
    "measure",
    "source_publisher",
    "source_title",
    "edition",
    "url",
    "accessed",
    "note",
)
#: Columns every row with an official figure must fill.
SOURCE_COLUMNS = RECEIPT_SOURCE_COLUMNS


VAT_IN_SAMPLE_CAVEAT = (
    "In-sample check: the firm generator is calibrated to HMRC VAT liability "
    "by turnover band and sector, so agreement here does not validate the model."
)

PAYBILL_ASSUMED_CAVEAT = (
    "Assumed input, not a validation: firm_paybill is not observed. It is the "
    "generator's headcount times the ONS ASHE 2025 section mean annual gross "
    "pay (all employee jobs), with working proprietors counted as employees "
    "and out-of-frame zero-turnover traders included, so agreement or "
    "disagreement here reflects that assumption, not the model."
)

PAY_SPLIT_UNSET_CAVEAT = (
    "The pay-distribution inputs (firm_employees_below_secondary_threshold, "
    "firm_paybill_below_secondary_threshold) are unset, so policyengine-uk "
    "takes the Class 1 base as paybill less headcount times the secondary "
    "threshold (an average-pay approximation), which understates the base "
    "where some staff earn below the threshold."
)

#: What policyengine-uk's variable measures, and how it differs from the
#: official measure in the receipts CSV. Read with the CSV's ``measure``.
MODEL_CAVEATS: Mapping[str, str] = {
    "firm_vat": (
        VAT_IN_SAMPLE_CAVEAT
        + " Model: accrued VAT liability of generated firms; official: HMRC cash "
        "receipts for all VAT, whose timing and coverage differ from the "
        "liability of firms in the ONS frame."
    ),
    "firm_employer_ni": (
        PAYBILL_ASSUMED_CAVEAT
        + " "
        + PAY_SPLIT_UNSET_CAVEAT
        + " Model: employer Class 1 secondary NICs after the Employment "
        "Allowance, accrued; official: cash, and also includes Class 1A "
        "(benefits in kind), so the official figure is larger in concept. "
        "Employment Allowance: firm_is_single_director_company is unset, so no "
        "firm is excluded as a single-director company and the allowance is "
        "overstated. Employers outside the generated VAT/PAYE frame (e.g. "
        "households employing domestic staff) pay into the official total."
    ),
    "apprenticeship_levy": (
        PAYBILL_ASSUMED_CAVEAT
        + " Model: accrued levy liability of generated firms; official: cash "
        "receipts from all employers, including those outside the generated "
        "VAT/PAYE frame."
    ),
    "corporation_tax": (
        "Model: onshore CT after the R&D expenditure credit, accrued, excluding "
        "the bank surcharge; official: onshore cash receipts gross of provisional "
        "tax credits (offshore ring-fence CT removed, Bank Surcharge and Bank Levy "
        "outside), with instalment timing that spans financial years."
    ),
    "bank_surcharge": (
        "Model: surcharge on taxable profit above the allowance, accrued; "
        "official: cash receipts. Bank Levy is not compared."
    ),
    "firm_business_rates": (
        "England only on both sides: the model gives 0 outside England. Model: "
        "gross rates less reliefs, one aggregate property per firm; official: "
        "NNDR3 gross rates payable less reliefs, before transitional "
        "arrangements, and including non-firm ratepayers (e.g. public bodies)."
    ),
}


#: Taxes computed from the assumed firm_paybill.
PAYBILL_TAXES = frozenset({"firm_employer_ni", "apprenticeship_levy"})


def read_receipts(path: Path) -> dict[str, Receipt]:
    """Read the receipts CSV, refusing missing columns or unsourced figures."""

    return _read_receipts(path, name_column="tax", value_column="official_value_gbp")


def financial_year(period: int) -> str:
    """2024 -> "2024-25", the UK financial year policyengine-uk's period starts."""

    return f"{period}-{(period + 1) % 100:02d}"


def format_gbp(value: float) -> str:
    """GBP in billions with a sign, e.g. -1.23bn -> "-£1.23bn"."""

    sign = "-" if value < 0 else ""
    return f"{sign}£{abs(value) / 1e9:,.2f}bn"


def _reasons(tax: str, receipt: Receipt | None) -> list[str]:
    """Why the model total and the official figure may differ, in order."""

    reasons = []
    if receipt is not None and receipt.measure:
        reasons.append(f"Official measure: {receipt.measure}.")
    if tax in MODEL_CAVEATS:
        reasons.append(MODEL_CAVEATS[tax])
    if tax == "firm_vat" and VAT_IN_SAMPLE_CAVEAT not in " ".join(reasons):
        reasons.insert(0, VAT_IN_SAMPLE_CAVEAT)
    if tax in PAYBILL_TAXES and PAYBILL_ASSUMED_CAVEAT not in " ".join(reasons):
        reasons.insert(0, PAYBILL_ASSUMED_CAVEAT)
    return reasons


def compare_tax(
    tax: str,
    *,
    model_total: float | None,
    populated: bool,
    receipt: Receipt | None,
) -> ComparisonRow:
    """One table row. No gap or percentage unless both sides are real figures."""

    return compare_total(
        tax,
        model_total=model_total,
        populated=populated,
        official_value=None if receipt is None else receipt.official_value,
        reasons=_reasons(tax, receipt),
        format_value=format_gbp,
    )


def build_rows(
    taxes: Mapping[str, Any], receipts: Mapping[str, Receipt]
) -> list[ComparisonRow]:
    """Rows in receipts order, then any model-only taxes."""

    return build_comparison_rows(
        taxes.get("totals", {}),
        taxes.get("inputs_populated", {}),
        receipts,
        reasons_for=_reasons,
        format_value=format_gbp,
    )


def render_markdown(
    rows: Sequence[ComparisonRow],
    receipts: Mapping[str, Receipt],
    *,
    period: str,
    note: str | None = None,
) -> str:
    return render_comparison_markdown(
        rows,
        receipts,
        title=f"UK firm taxes against official receipts, {period}",
        name_header="Tax",
        note=note,
    )


def compare(taxes: Mapping[str, Any], receipts: Mapping[str, Receipt]) -> str:
    """The markdown report for a taxes JSON object and the receipts."""

    period = financial_year(int(taxes["period"]))
    periods = {r.period for r in receipts.values()}
    if periods != {period}:
        raise ValueError(
            f"model period {period} does not match receipts {sorted(periods)}"
        )
    return render_markdown(
        build_rows(taxes, receipts), receipts, period=period, note=taxes.get("note")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--taxes-json", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args(argv)

    taxes = json.loads(args.taxes_json.read_text(encoding="utf-8"))
    report = compare(taxes, read_receipts(args.receipts))
    sys.stdout.write(report)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
