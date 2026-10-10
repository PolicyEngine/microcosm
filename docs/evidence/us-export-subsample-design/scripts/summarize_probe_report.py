"""Summarize a ``probe_report.json`` written with ``--reference-release-dir``.

Prints, as Markdown:

1. every verdict, by stage, with its authority;
2. each reform-coverage smoke probe against the full-size build: the
   subsample effect, the full-size effect, their relative difference, the z
   of that difference in the subsample's standard errors, and the effective
   households the standard error rests on;
3. the z distribution and how the authoritative verdicts compare.

Standard library only, so it runs anywhere the report does.

Usage::

    python summarize_probe_report.py <probe_report.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _money(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value / 1e6:,.1f}"


def _number(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def verdict_table(report: dict) -> list[str]:
    lines = [
        "| Stage | Check | Verdict | Authority |",
        "|---|---|---|---|",
    ]
    for verdict in report["verdicts"]:
        if verdict["stage"] == "reform_coverage_smoke":
            continue
        lines.append(
            f"| {verdict['stage']} | {verdict['check']} | {verdict['verdict']} "
            f"| {verdict['authority']} |"
        )
    return lines


def smoke_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Probe | Passes | Authority | Effect ($M) | Full size ($M) "
        "| Relative difference | z | Effective households |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda row: row["probe"]):
        reference = row.get("reference") or {}
        relative = reference.get("relative_difference")
        lines.append(
            f"| `{row['probe']}` | {'yes' if row['passed'] else '**no**'} "
            f"| {row['authority']}{' (take-all)' if row['take_all'] else ''} "
            f"| {_money(row['effect'])} | {_money(reference.get('effect'))} "
            f"| {'' if relative is None else f'{relative:+.1%}'} "
            f"| {_number(reference.get('z'))} "
            f"| {_number(row.get('effective_variance_households'), 1)} |"
        )
    return lines


def _reference(row: dict) -> dict:
    return row.get("reference") or {}


def smoke_summary(rows: list[dict]) -> list[str]:
    """Counts and the z distribution; a report in which no probe has a
    z-score (every standard error zero or missing) says so."""
    scored = [row for row in rows if _reference(row).get("z") is not None]
    magnitudes = [abs(_reference(row)["z"]) for row in scored]
    authoritative = [row for row in rows if row["authority"] == "authoritative"]
    disagree = [
        row["probe"] for row in rows if _reference(row).get("verdict_agrees") is False
    ]
    authoritative_z = [
        abs(_reference(row)["z"])
        for row in authoritative
        if _reference(row).get("z") is not None
    ]
    if scored:
        worst = max(scored, key=lambda row: abs(_reference(row)["z"]))
        z_line = (
            f"- with a z-score: {len(scored)}; median |z| "
            f"{statistics.median(magnitudes):.2f}; largest |z| "
            f"{abs(_reference(worst)['z']):.2f} (`{worst['probe']}`, "
            f"{worst['authority']}); |z| > 3: "
            f"{sum(value > 3 for value in magnitudes)}; |z| > 4: "
            f"{sum(value > 4 for value in magnitudes)}"
        )
    else:
        z_line = "- with a z-score: 0 (every standard error is zero or missing)"
    exact = [
        row["probe"]
        for row in authoritative
        if row["take_all"]
        and _reference(row).get("effect") is not None
        and row["effect"] == _reference(row)["effect"]
    ]
    return [
        f"- probes: {len(rows)}; passing: {sum(row['passed'] for row in rows)}; "
        f"authoritative: {len(authoritative)} "
        f"(passing: {sum(row['passed'] for row in authoritative)})",
        z_line,
        f"- largest |z| among authoritative verdicts: {max(authoritative_z):.2f}"
        if authoritative_z
        else "",
        "- authoritative take-all probes equal to the full-size effect exactly: "
        + (", ".join(f"`{probe}`" for probe in exact) or "none"),
        "- verdicts that disagree with the full-size build: "
        + (", ".join(f"`{probe}`" for probe in disagree) or "none"),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    rows = report["stages"]["reform_coverage_smoke"]["probes"]
    summary = report["summary"]
    print("## Summary\n")
    print(
        f"- authoritative release failures: "
        f"{len(summary['authoritative_failures'])}; informational failures: "
        f"{len(summary['informational_failures'])}; probe failures: "
        f"{len(summary.get('probe_failures', []))}\n"
    )
    print("## Verdicts outside the smoke\n")
    print("\n".join(verdict_table(report)) + "\n")
    print("## Smoke probes against the full-size build\n")
    print("\n".join(line for line in smoke_summary(rows) if line) + "\n")
    print("\n".join(smoke_table(rows)))


if __name__ == "__main__":
    main()
