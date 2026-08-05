"""Build the synthetic US import-entry file from the P1 margins (populace#615 P2).

Reads the P1 margin artifacts (``margins_hts10_country_month.parquet`` plus
the archived CBP statistics page) and generates the **synthetic** weighted
entry file for the pilot window, reproducing every calibrated
HTS-10 x country x month customs-value margin exactly. Emits:

- ``synthetic_import_entries.parquet`` — the entries + weights table, with
  dataset-level synthetic/provenance metadata on the parquet schema,
- ``assumptions.json`` — the assumptions register (size model, anchors,
  solved sigma, achieved statistics, known gaps),
- ``validation_report.json`` — exact-margin check results plus context
  statistics (achieved vs anchor moments, weighted entry counts).

Everything in the output is generated, never observed; the file is a
weighted representation whose only calibrated facts are the margins it
reproduces. Exit code 1 on any validation failure.

Example::

    uv run python tools/build_us_import_entries.py \
        --margins-dir ~/runtime/out/us-import-entry-margins \
        --start 2026-01 --out-dir ~/runtime/out/us-import-entries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.us_runtime.us_trade.cbp_entry_stats import (  # noqa: E402
    parse_cbp_trade_stats,
)
from populace.build.us_runtime.us_trade.entry_generator import (  # noqa: E402
    DEFAULT_INFORMAL_VALUE_THRESHOLD,
    DEFAULT_STRATA,
    assumption_to_json,
    generate_entries,
    solve_size_assumption,
    validate_entries_against_margins,
)

#: The composed tariff spine (rulespec-us us:policies/cbp/us-tariff-duty/
#: composition) takes effect 2026-02-15; entries dated before then are
#: outside the engine domain (P4 runs use the 2026-02+ subset).
ENGINE_DOMAIN_START = "2026-02-15"

CBP_ARCHIVE_NAME = "cbp_newsroom_stats_trade.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--margins-dir",
        required=True,
        type=Path,
        help="P1 output directory (margins parquet + archived CBP page).",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start", default="2026-01", help="Pilot window start.")
    parser.add_argument(
        "--end", default=None, help="Pilot window end; default = latest margin month."
    )
    parser.add_argument("--strata", type=int, default=DEFAULT_STRATA)
    parser.add_argument(
        "--informal-threshold",
        type=int,
        default=DEFAULT_INFORMAL_VALUE_THRESHOLD,
        help="Informal-entry value threshold in USD (19 CFR 143.21).",
    )
    parser.add_argument(
        "--mean-entry-value",
        type=float,
        default=None,
        help="Override the CBP-derived mean entry value anchor.",
    )
    parser.add_argument(
        "--informal-share",
        type=float,
        default=None,
        help="Override the CBP-derived informal count share anchor.",
    )
    args = parser.parse_args()

    margins_path = args.margins_dir / "margins_hts10_country_month.parquet"
    margins = pd.read_parquet(margins_path)
    end = args.end or str(margins["period"].max())
    window = margins.loc[
        (margins["period"] >= args.start) & (margins["period"] <= end)
    ].reset_index(drop=True)
    if window.empty:
        print(f"[entries] FATAL: no margin cells in {args.start}..{end}")
        return 1
    months = sorted(window["period"].unique())
    print(f"[entries] window {months[0]} .. {months[-1]} ({len(window)} margin rows)")

    anchors = _anchors(
        args.margins_dir,
        end_month=end,
        mean_override=args.mean_entry_value,
        share_override=args.informal_share,
    )
    print(
        f"[entries] anchors: mean entry value "
        f"{anchors['mean_entry_value']:,.0f} USD, informal share "
        f"{anchors['informal_count_share']:.4f} ({anchors['basis']})"
    )

    cells = window.loc[window["con_val_mo"] > 0]
    assumption = solve_size_assumption(
        cells["con_val_mo"].to_numpy(),
        mean_entry_value=anchors["mean_entry_value"],
        informal_count_share=anchors["informal_count_share"],
        informal_value_threshold=args.informal_threshold,
        strata=args.strata,
        anchor_provenance=anchors,
    )
    print(
        f"[entries] solved sigma={assumption.sigma:.4f}; achieved informal "
        f"share {assumption.achieved_informal_count_share:.4f}, achieved "
        f"mean {assumption.achieved_mean_entry_value:,.0f} USD, "
        f"{assumption.total_weighted_entries:,} weighted entries"
    )

    entries = generate_entries(window, assumption)
    report = validate_entries_against_margins(entries, window)
    cross = _chapter_country_cross_check(entries, window)
    report.update(cross)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    generator_block = {
        "producer": "populace.build.us_runtime.us_trade.entry_generator",
        "issue": "PolicyEngine/populace#615",
        "phase": "P2",
        "synthetic": True,
        "generated_at": generated_at,
        "window": {"start": months[0], "end": months[-1]},
        "margins_parquet_sha256": _sha256(margins_path),
        "engine_domain_start": ENGINE_DOMAIN_START,
    }
    entries_path = out_dir / "synthetic_import_entries.parquet"
    _write_labeled_parquet(entries, entries_path, assumption, generator_block)

    (out_dir / "assumptions.json").write_text(assumption_to_json(assumption))
    report.update(
        {
            "window": {"start": months[0], "end": months[-1], "months": len(months)},
            "entries_parquet_sha256": _sha256(entries_path),
            "generated_at": generated_at,
            "monthly_weighted_entries": _monthly_counts(entries),
            "engine_domain_start": ENGINE_DOMAIN_START,
            "engine_runnable_months": [
                month for month in months if f"{month}-15" >= ENGINE_DOMAIN_START
            ],
        }
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _anchors(
    margins_dir: Path,
    *,
    end_month: str,
    mean_override: float | None,
    share_override: float | None,
) -> dict:
    """CBP fiscal-year anchors from the P1 archived page, or overrides."""
    if mean_override is not None and share_override is not None:
        return {
            "basis": "explicit overrides",
            "mean_entry_value": float(mean_override),
            "informal_count_share": float(share_override),
        }
    archive = margins_dir / CBP_ARCHIVE_NAME
    raw = archive.read_bytes()
    stats = parse_cbp_trade_stats(raw)
    year, month = int(end_month[:4]), int(end_month[5:7])
    fiscal_year = year + 1 if month >= 10 else year
    cells = {
        (cell.measure_id, cell.fiscal_year): cell.exact_value
        for cell in stats.exact_cells()
    }
    try:
        total_value = cells[("total_import_value", fiscal_year)]
        total_entries = cells[("total_entry_summaries", fiscal_year)]
        informal = cells[("informal_entry_summaries", fiscal_year)]
    except KeyError as error:
        raise SystemExit(
            f"CBP archive has no exact FY{fiscal_year} cell for {error}; "
            "pass --mean-entry-value/--informal-share explicitly."
        ) from None
    anchors = {
        "basis": (
            f"CBP Trade Statistics FY{fiscal_year} published totals "
            "(exact cells parsed from the archived page)"
        ),
        "source_url": "https://www.cbp.gov/newsroom/stats/trade",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "publisher_as_of_note": stats.as_of_note,
        "fiscal_year": fiscal_year,
        "total_import_value": total_value,
        "total_entry_summaries": total_entries,
        "informal_entry_summaries": informal,
        "mean_entry_value": total_value / total_entries,
        "informal_count_share": informal / total_entries,
    }
    if mean_override is not None:
        anchors["mean_entry_value"] = float(mean_override)
        anchors["basis"] += "; mean overridden"
    if share_override is not None:
        anchors["informal_count_share"] = float(share_override)
        anchors["basis"] += "; informal share overridden"
    return anchors


def _chapter_country_cross_check(entries: pd.DataFrame, margins: pd.DataFrame) -> dict:
    """Independent re-aggregation on the P3 dashboard axis (exact)."""
    produced = (
        (entries["weight"] * entries["customs_value"])
        .groupby(
            [
                entries["period"].astype(str),
                entries["chapter"].astype(str),
                entries["census_country_code"].astype(str),
            ]
        )
        .sum()
        .sort_index()
    )
    cells = margins.loc[margins["con_val_mo"] > 0]
    expected = (
        cells.groupby(
            [
                cells["period"].astype(str),
                cells["hts10"].str[:2],
                cells["cty_code"].astype(str),
            ]
        )["con_val_mo"]
        .sum()
        .sort_index()
    )
    if (
        len(produced) != len(expected)
        or not (produced.to_numpy() == expected.to_numpy()).all()
    ):
        raise ValueError("Chapter x country re-aggregation does not match margins.")
    return {"chapter_country_cells_exact": int(len(expected))}


def _write_labeled_parquet(
    entries: pd.DataFrame,
    path: Path,
    assumption,
    generator_block: dict,
) -> None:
    table = pa.Table.from_pandas(entries, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"populace_us_trade.synthetic": b"true",
            b"populace_us_trade.dataset": b"synthetic_import_entries",
            b"populace_us_trade.generator": json.dumps(
                generator_block, sort_keys=True
            ).encode("utf-8"),
            b"populace_us_trade.assumptions": assumption_to_json(assumption).encode(
                "utf-8"
            ),
        }
    )
    pq.write_table(table.replace_schema_metadata(metadata), path, compression="zstd")


def _monthly_counts(entries: pd.DataFrame) -> dict:
    counts = entries.groupby(entries["period"].astype(str))["weight"].sum()
    return {month: int(count) for month, count in counts.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
