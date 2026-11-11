"""Extract the pinned MSD rates page from local source files.

Set NZ_MSD_RATES_SOURCES to the staged source directory. The generator checks
the receipt against the pin in this file before it interprets any table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "packages/microcosm-build/src/microcosm/build/nz/msd_benefit_rates_20260401.json"
)
PAGE = "benefit-rates-april-2026.html"
PIN = {
    "url": "https://www.workandincome.govt.nz/products/benefit-rates/benefit-rates-april-2026.html",
    "sha256": "41817fc057388e06997657a786774015d7e9b74e98ca3104fa860ad90e31a80c",
    "retrieved_at": "2026-10-24T18:08:25Z",
    "licence": "CC BY 4.0 (Crown copyright, MSD; per /about-this-site/privacy-disclaimer-and-copyright-information)",
}
PROGRAMMES = {
    "Jobseeker Support": "jobseeker_support",
    "Sole Parent Support": "sole_parent_support",
    "Supported Living Payment": "supported_living_payment",
}


class _Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.heading = self.subheading = ""
        self.capture = None
        self.text = []
        self.table = None
        self.row = []

    def handle_starttag(self, tag, attrs):
        if tag in {"h2", "h3", "td", "th"}:
            self.capture, self.text = tag, []
        elif tag == "table":
            self.table = [self.heading, self.subheading, []]
        elif tag == "tr":
            self.row = []

    def handle_data(self, data):
        if self.capture:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == self.capture:
            value = " ".join("".join(self.text).split())
            if tag == "h2":
                self.heading, self.subheading = value, ""
            elif tag == "h3":
                self.subheading = value
            elif self.table is not None:
                self.row.append(value)
            self.capture = None
        elif tag == "tr" and self.table is not None and self.row:
            self.table[2].append(self.row)
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _money(value):
    text = value.removeprefix("$").replace(",", "")
    if re.fullmatch(r"[0-9]+\.[0-9]{2}", text) is None:
        raise ValueError(f"Expected a published decimal amount, got {value!r}")
    return text


def generate(source_dir):
    """Return all published pairs and the 14 individual validation points."""
    source_dir = Path(source_dir)
    (receipt,) = [
        json.loads(line)
        for line in (source_dir / "receipts.jsonl").read_text().splitlines()
        if line
    ]
    for key, expected in PIN.items():
        source_key = "fetched_at" if key == "retrieved_at" else key
        if receipt.get(source_key) != expected:
            raise ValueError(f"MSD pinned receipt mismatch: {key}")
    page = (source_dir / PAGE).read_bytes()
    if hashlib.sha256(page).hexdigest() != PIN["sha256"]:
        raise ValueError("MSD page sha256 mismatch")
    parser = _Tables()
    parser.feed(page.decode("utf-8"))
    weekly, fortnightly = [], []
    for heading, section, rows in parser.tables:
        if heading in PROGRAMMES:
            programme = PROGRAMMES[heading]
            for row in rows[1:]:
                if programme == "sole_parent_support":
                    category, net, gross = "Sole parent", *row[:2]
                else:
                    category, net, gross = row[:3]
                validation = "(total)" not in category
                if "1 partner not on benefit" in category:
                    validation = programme == "supported_living_payment"
                if "1 partner on NZ Super" in category:
                    validation = False
                weekly.append(
                    {
                        "programme": programme,
                        "section": section,
                        "category": category,
                        "after_tax_at_m": _money(net),
                        "before_tax": _money(gross),
                        "individual_validation_point": validation,
                    }
                )
        elif heading == "NZ Super and Veteran's Pension":
            for row in rows[1:]:
                if len(row) != 4 or not row[1].startswith("$"):
                    continue
                category, at_m, at_s, gross = row
                fortnightly.append(
                    {
                        "section": section,
                        "category": category,
                        "after_tax_at_m": _money(at_m),
                        "after_tax_at_s": None if at_s == "N/A" else _money(at_s),
                        "before_tax": _money(gross),
                        "diagnostic_only": True,
                    }
                )
    if (
        len(weekly) != 21
        or sum(row["individual_validation_point"] for row in weekly) != 14
        or len(fortnightly) != 10
    ):
        raise ValueError("MSD pinned table roster changed")
    return {
        "schema_version": 1,
        "source": PIN,
        "weekly_main_benefits": weekly,
        "fortnightly_nz_super": fortnightly,
        "notes": "This resource retains MSD's published couple totals; the inverse validation uses the 14 individual rate points. NZ Super supplies diagnostic comparisons only; the gross-up never inverts it.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=os.environ.get("NZ_MSD_RATES_SOURCES"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.sources:
        parser.error("--sources or NZ_MSD_RATES_SOURCES is required")
    rendered = json.dumps(generate(args.sources), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.out.read_text() != rendered:
            raise SystemExit(f"Stale generated MSD rates: {args.out}")
        print("MSD pinned rates: current")
    else:
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
