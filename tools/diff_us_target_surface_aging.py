"""Diff the US target surface un-aged vs aged from one Ledger feed.

The review harness behind PolicyEngine/microcosm#287: compiles the identical
target surface twice from the same consumer-fact feed — pre-contract behavior
(un-aged, waived) and the release default (aged under cbo_growth_factor_aging)
— and reports what calibration would be asked to hit. Because calibration
hits targets near-exactly (#212), target-level deltas approximate the next
build's aggregate deltas.

Usage:
    uv run python tools/diff_us_target_surface_aging.py <feed> [--period 2024]

``<feed>`` is a Ledger consumer-artifact directory or consumer_facts.jsonl.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.fiscal_targets import (
    compile_us_fiscal_target_registry,
)

HEADLINE_KEYS = (
    "income_tax_liability",
    "adjusted_gross_income",
    "eitc",
    "earned_income",
    "child_tax_credit",
    "itemized",
    "state_and_local",
    "wages_salaries",
    "taxable_interest",
    "qualified_business",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feed", help="Consumer artifact directory or JSONL file")
    parser.add_argument("--period", type=int, default=2024)
    args = parser.parse_args()

    artifact = load_ledger_consumer_artifact(args.feed)
    congressional_district_vintage_crosswalk = (
        load_congressional_district_vintage_crosswalk(
            default_congressional_district_vintage_crosswalk_path()
        )
    )
    print(
        f"feed: {artifact.path.name} rows={artifact.fact_row_count} "
        f"facts_sha256={artifact.facts_sha256[:12]}…"
    )

    unaged = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=args.period,
        allow_unaged_dollar_targets=True,
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
    )
    aged = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=args.period,
        age_targets=True,
        allow_unaged_dollar_targets=True,
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
    )

    before_by = {spec.name: spec for spec in unaged.specs}
    after_by = {spec.name: spec for spec in aged.specs}
    assert set(before_by) == set(after_by), "aging must not add or remove targets"

    rows = []
    for name, spec in after_by.items():
        meta = spec.metadata
        rows.append(
            {
                "name": name,
                "family": spec.family,
                "before": before_by[name].value,
                "after": spec.value,
                "factor": float(meta.get("aging_factor", "1") or 1),
                "basis": meta.get("basis", ""),
                "factor_source": meta.get("aging_factor_source", ""),
                "waived": bool(meta.get("period_contract_waiver")),
                "source_period": meta.get("source_period", "")
                or meta.get("ledger_fact_period", ""),
            }
        )

    aged_rows = [r for r in rows if r["basis"] == "projection"]
    waived_rows = [r for r in rows if r["waived"]]
    untouched = len(rows) - len(aged_rows) - len(waived_rows)
    print(
        f"\ntargets: {len(rows)} | aged: {len(aged_rows)} | "
        f"waived: {len(waived_rows)} | untouched: {untouched}"
    )

    by_family = defaultdict(lambda: [0.0, 0.0, 0])
    for r in aged_rows:
        by_family[r["family"]][0] += r["before"]
        by_family[r["family"]][1] += r["after"]
        by_family[r["family"]][2] += 1
    print("\n== aged dollars by family ==")
    for family, (b, a, n) in sorted(
        by_family.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])
    ):
        pct = (a / b - 1) * 100 if b else float("nan")
        print(
            f"  {family:26s} n={n:5d} {b / 1e9:12,.1f}B → {a / 1e9:12,.1f}B ({pct:+.2f}%)"
        )

    print("\n== headline rows ==")
    shown = 0
    for r in sorted(aged_rows, key=lambda r: -abs(r["after"] - r["before"])):
        if not any(key in r["name"] for key in HEADLINE_KEYS):
            continue
        print(
            f"  {r['name'][:74]:76s} {r['before'] / 1e9:10,.1f}B → "
            f"{r['after'] / 1e9:10,.1f}B ×{r['factor']:.4f} "
            f"[{r['source_period']}→{args.period}]"
        )
        shown += 1
        if shown >= 14:
            break

    print("\n== factors in play ==")
    factor_counts = Counter(
        (round(r["factor"], 4), r["factor_source"]) for r in aged_rows
    )
    for (factor, source), n in factor_counts.most_common(10):
        print(f"  ×{factor:.4f} n={n:5d} {source}")

    if waived_rows:
        print("\n== waived rows (factor gaps) ==")
        gaps = Counter((r["family"], r["source_period"]) for r in waived_rows)
        for (family, period), n in gaps.most_common(10):
            print(f"  {family:26s} fact_period={period:8s} n={n}")

    total_before = sum(r["before"] for r in aged_rows)
    total_after = sum(r["after"] for r in aged_rows)
    if total_before:
        print(
            f"\nTOTAL aged-dollar surface: {total_before / 1e9:,.1f}B → "
            f"{total_after / 1e9:,.1f}B ({(total_after / total_before - 1) * 100:+.2f}%)"
        )


if __name__ == "__main__":
    main()
