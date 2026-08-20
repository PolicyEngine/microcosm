"""Build disclosure-safe WAS wealth support bounds from a donor-like CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.was_wealth import (
    UK_WAS_WEALTH_OUTPUT_COLUMNS,
    clean_was_household_table,
    donor_realized_ranges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "packages/microcosm-build/src/microcosm/build/uk/was_wealth_support_bounds.json"
)


def build_support_bounds(tab_path: Path) -> dict[str, object]:
    tab_bytes = tab_path.read_bytes()
    tab_sha256 = hashlib.sha256(tab_bytes).hexdigest()
    raw = pd.read_csv(tab_path, sep="\t")
    donor = clean_was_household_table(raw)
    exact = donor_realized_ranges(donor)
    return {
        "version": 1,
        "country": "uk",
        "policy": (
            "Disclosure-safe outward-rounded WAS wealth support bounds for "
            "the UK support gate, generated from the donor household tab "
            "recorded under source.tab_sha256 by "
            "tools/build_uk_was_wealth_support_bounds.py. Values are "
            "donor-realized exact ranges rounded outward to one significant "
            "figure, so unit-record minima and maxima are never committed."
        ),
        "source": {
            "ukds_study_number": 7215,
            "doi": "10.5255/UKDA-SN-7215-20",
            "artifact": "was_round_8_hhold_eul_may_2025_230525.tab",
            "tab_sha256": tab_sha256,
            "sdc_treatment": (
                "Exact donor min/max values are rounded outward to one "
                "significant figure before commit, so no committed number "
                "is an exact unit-record value. Caveat considered and "
                "accepted: WAS applies top-coding and a round top-code can "
                "survive outward rounding unchanged; top-codes are "
                "published survey methodology rather than unit-record "
                "disclosures. Adjudicated acceptable under the UKDS EUL "
                "disclosure rules by juaristi22 on 2026-08-19 "
                "(microcosm#714)."
            ),
        },
        "bounds": {
            column: list(_outward_round(bounds))
            for column, bounds in exact.items()
            if column in UK_WAS_WEALTH_OUTPUT_COLUMNS
        },
        "chronicle": [
            "Support bounds generated from the tab pinned as "
            f"source.tab_sha256 ({tab_sha256[:12]}…) with outward SDC "
            "rounding."
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--was-tab", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build_support_bounds(args.was_tab)
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.check:
        if args.output_json.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output_json} is stale.")
        return 0
    args.output_json.write_text(rendered, encoding="utf-8")
    return 0


def _outward_round(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = bounds
    return (_round_down(lo), _round_up(hi))


def _round_down(value: float) -> float:
    if value == 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(abs(value)))
    return math.floor(value / magnitude) * magnitude


def _round_up(value: float) -> float:
    if value == 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(abs(value)))
    return math.ceil(value / magnitude) * magnitude


if __name__ == "__main__":
    raise SystemExit(main())
