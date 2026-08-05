"""Cross-check bulk-built import margins against the Census API (populace#615 P1).

The margins artifacts are built from the monthly bulk IMDB archives
(``tools/build_us_import_entry_margins.py``). This tool re-fetches a
stratified sample of (month, chapter) pairs over the Census International
Trade API — the second official channel for the same statistical series —
and compares every HTS10 × country cell in the sample exact-integer on the
four dollar measures, plus the publisher per-commodity totals.

Two channels serving the same publication should agree cell-for-cell when
their revision vintages match; any divergence is reported per cell so
revision skew (value-level, scattered, month-concentrated) can be told
apart from parse defects (structural, systematic). Quantity columns are
compared only where the API publishes a value (the API omits quantities on
some lines; the bulk files publish integer zeros) and quantity differences
are reported separately, never gating.

The default sample covers the API-500-prone giant chapters (84/85/87),
mid and small chapters, and months across both statistical years. API
responses cache under ``--cache-dir`` (the lane-G cache is reusable), so
re-runs are free.

Example::

    CENSUS_API_KEY=... uv run python tools/crosscheck_us_import_margins_api.py \
        --margins out/us-import-entry-margins/margins_hts10_country_month.parquet \
        --totals out/us-import-entry-margins/census_totals_hts10_month.parquet \
        --cache-dir ~/.cache/populace/us-trade/census-imports-hs \
        --out out/us-import-entry-margins/crosscheck_api_report.json

Exit code: 1 on any dollar-measure mismatch (report still written), 0 on
full agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.us_runtime.us_trade import fetch_imports_month  # noqa: E402

#: Stratified default sample: (month, chapters). Giants 84/85/87 exercise
#: the prefix-split path; 01/02/41 are small; 30/61/90/94/98 mid-size; both
#: statistical years and window endpoints are covered.
DEFAULT_SAMPLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("2025-01", ("01", "30", "85", "87")),
    ("2025-08", ("41", "61", "84", "98")),
    ("2026-01", ("02", "44", "85", "90")),
    ("2026-05", ("03", "84", "87", "94")),
)

_DOLLAR_MEASURES = ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo")
_QUANTITY_MEASURES = ("con_qy1_mo", "gen_qy1_mo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margins", required=True, type=Path)
    parser.add_argument("--totals", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--census-key", default=None, help="Census API key; default $CENSUS_API_KEY."
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        metavar="YYYY-MM:CH,CH,...",
        help="Override the sample; repeatable (e.g. --pair 2025-01:85,87).",
    )
    args = parser.parse_args()

    api_key = args.census_key or os.environ.get("CENSUS_API_KEY")
    if not api_key:
        parser.error("A Census API key is required (--census-key or CENSUS_API_KEY).")

    sample = _parse_sample(args.pair) if args.pair else DEFAULT_SAMPLE
    margins = pd.read_parquet(args.margins)
    totals = pd.read_parquet(args.totals)

    pair_reports: list[dict[str, object]] = []
    for month, chapters in sample:
        for chapter in chapters:
            pair_reports.append(
                _compare_pair(month, chapter, margins, totals, api_key, args.cache_dir)
            )

    def _total(field: str) -> int:
        return int(sum(int(pair[field]) for pair in pair_reports))  # type: ignore[arg-type]

    gating_failures = (
        _total("dollar_mismatch_cells")
        + _total("total_mismatches")
        + _total("api_reconciliation_failures")
    )
    report = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sample_pairs": len(pair_reports),
        "cells_compared": _total("cells_compared"),
        "dollar_measures": list(_DOLLAR_MEASURES),
        "dollar_mismatch_cells": _total("dollar_mismatch_cells"),
        "publisher_total_mismatches": _total("total_mismatches"),
        "api_reconciliation_failures": _total("api_reconciliation_failures"),
        "api_totals_absent_pairs": _total("api_totals_absent"),
        "gating_failures": gating_failures,
        "pairs": pair_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "pairs"},
            indent=2,
            sort_keys=True,
        )
    )
    for pair in pair_reports:
        print(
            f"[crosscheck] {pair['month']} ch{pair['chapter']}: "
            f"{pair['cells_compared']} cells, "
            f"{pair['dollar_mismatch_cells']} dollar mismatches, "
            f"{pair['total_mismatches']} publisher-total mismatches, "
            f"{pair['api_reconciliation_failures']} API reconciliation "
            f"failures, api_only={pair['api_only_cells']} "
            f"bulk_only={pair['bulk_only_cells']}"
        )
    return 1 if gating_failures else 0


def _compare_pair(
    month: str,
    chapter: str,
    margins: pd.DataFrame,
    totals: pd.DataFrame,
    api_key: str,
    cache_dir: Path,
    *,
    fetch: object | None = None,
) -> dict[str, object]:
    pulled = fetch_imports_month(
        month,
        api_key,
        cache_dir=cache_dir,
        chapters=(chapter,),
        max_workers=1,
        fetch=fetch,
    )
    # The API leg runs its own exact reconciliation against the publisher's
    # '-' totals; a pair whose API pull is internally inconsistent must
    # never count as agreement.
    api_reconciliation_failures = len(pulled.reconciliation_failures)
    api_cells = pd.DataFrame(list(pulled.country_rows))
    api_totals = pd.DataFrame(list(pulled.total_rows))

    bulk_cells = margins.loc[
        (margins["period"] == month) & (margins["chapter"] == chapter)
    ]
    bulk_totals = totals.loc[
        (totals["period"] == month) & (totals["hts10"].str.startswith(chapter))
    ]

    # Publisher per-commodity totals: the bulk control file is a YTD union
    # (all-zero rows for commodities inactive this month); the API only
    # publishes totals for active commodities. Compare on active rows.
    active_bulk_totals = bulk_totals.loc[
        bulk_totals[list(_DOLLAR_MEASURES)].any(axis="columns")
    ]

    if api_cells.empty:
        return {
            "month": month,
            "chapter": chapter,
            "cells_compared": 0,
            "dollar_mismatch_cells": int(len(bulk_cells)),
            "total_mismatches": int(len(active_bulk_totals)),
            "api_reconciliation_failures": api_reconciliation_failures,
            "api_totals_absent": 1,
            "api_only_cells": 0,
            "bulk_only_cells": int(len(bulk_cells)),
            "quantity_diff_cells": 0,
            "api_null_quantity_cells": 0,
            "note": "API returned no rows for this pair; bulk has rows.",
            "mismatches": [],
        }

    key = ["hts10", "cty_code"]
    api_indexed = api_cells.set_index(key).sort_index()
    bulk_indexed = bulk_cells.set_index(key).sort_index()
    joined = api_indexed.join(
        bulk_indexed, how="outer", lsuffix="_api", rsuffix="_bulk"
    )
    api_only = joined[joined[f"{_DOLLAR_MEASURES[0]}_bulk"].isna()]
    bulk_only = joined[joined[f"{_DOLLAR_MEASURES[0]}_api"].isna()]
    both = joined.dropna(
        subset=[f"{_DOLLAR_MEASURES[0]}_api", f"{_DOLLAR_MEASURES[0]}_bulk"]
    )

    mismatch_mask = pd.Series(False, index=both.index)
    mismatch_details: list[dict[str, object]] = []
    for measure in _DOLLAR_MEASURES:
        diff = both[f"{measure}_api"].astype("int64") != both[f"{measure}_bulk"].astype(
            "int64"
        )
        mismatch_mask |= diff
        for index_value in both.index[diff][:10]:
            row = both.loc[index_value]
            mismatch_details.append(
                {
                    "hts10": index_value[0],
                    "cty_code": index_value[1],
                    "measure": measure,
                    "api": int(row[f"{measure}_api"]),
                    "bulk": int(row[f"{measure}_bulk"]),
                }
            )
    # Unmatched cells are dollar mismatches by definition (value vs absent).
    dollar_mismatch_cells = int(mismatch_mask.sum()) + len(api_only) + len(bulk_only)
    for frame, side in ((api_only, "api_only"), (bulk_only, "bulk_only")):
        for index_value in frame.index[:10]:
            mismatch_details.append(
                {
                    "hts10": index_value[0],
                    "cty_code": index_value[1],
                    "measure": side,
                    "api": None,
                    "bulk": None,
                }
            )

    quantity_diff = 0
    api_null_quantity = 0
    for measure in _QUANTITY_MEASURES:
        api_column = both[f"{measure}_api"]
        bulk_column = both[f"{measure}_bulk"]
        published = api_column.notna()
        api_null_quantity += int((~published).sum())
        quantity_diff += int(
            (
                api_column[published].astype("int64")
                != bulk_column[published].astype("int64")
            ).sum()
        )

    total_mismatches = 0
    api_totals_absent = 0
    if api_totals.empty:
        # An active chapter with no API '-' rows means the totals leg has
        # no counterparty; report it rather than silently skipping.
        api_totals_absent = 1 if len(active_bulk_totals) else 0
    else:
        api_total_indexed = api_totals.set_index("hts10")
        bulk_total_indexed = active_bulk_totals.set_index("hts10")
        total_joined = api_total_indexed.join(
            bulk_total_indexed, how="outer", lsuffix="_api", rsuffix="_bulk"
        )
        one_sided = (
            total_joined[f"{_DOLLAR_MEASURES[0]}_api"].isna()
            | total_joined[f"{_DOLLAR_MEASURES[0]}_bulk"].isna()
        )
        total_mismatches += int(one_sided.sum())
        matched = total_joined[~one_sided]
        for measure in _DOLLAR_MEASURES:
            api_side = matched[f"{measure}_api"].astype("int64")
            bulk_side = matched[f"{measure}_bulk"].astype("int64")
            total_mismatches += int((api_side != bulk_side).sum())

    return {
        "month": month,
        "chapter": chapter,
        "cells_compared": int(len(both)),
        "dollar_mismatch_cells": dollar_mismatch_cells,
        "total_mismatches": total_mismatches,
        "api_reconciliation_failures": api_reconciliation_failures,
        "api_totals_absent": api_totals_absent,
        "api_only_cells": int(len(api_only)),
        "bulk_only_cells": int(len(bulk_only)),
        "quantity_diff_cells": quantity_diff,
        "api_null_quantity_cells": api_null_quantity,
        "mismatches": mismatch_details[:40],
    }


def _parse_sample(
    pairs: list[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for pair in pairs:
        month, _, chapter_list = pair.partition(":")
        chapters = tuple(
            chapter.strip() for chapter in chapter_list.split(",") if chapter.strip()
        )
        if not chapters:
            raise SystemExit(f"--pair {pair!r} carries no chapters.")
        parsed.append((month, chapters))
    return tuple(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
