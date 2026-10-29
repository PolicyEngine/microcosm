"""Taxpaying and non-taxpaying State Pension recipients HMRC implies (microcosm#1069).

SPI 2023-24 counts 7.98m taxpayers with a State Pension (Table 3.6, all
ranges, "Pension Income from National Insurance contributions (Number of
individuals)", in thousands to three figures). Over Table 2.1's 8.16m Income
Tax payers of State Pension age in 2023-24, that is 97.79%. Holding the share,
it is applied to the calendar-2025 window (1/4 of 2024-25 and 3/4 of 2025-26)
of two levels from the vendored ITL rows:

* the twelve region-tier areas of Table 2.2, the residents the frame holds;
* Table 2.1's United Kingdom row, which also counts taxpayers abroad or of
  unknown address.

The two give the range for taxpaying recipients. Non-taxpaying recipients are
the bound DWP and DfC State Pension recipients (from ``build_comparison.json``)
minus each end. Each build's measured split is set against the range.

    python3 hmrc_implied_recipients.py --vendored <hmrc_itl_taxpayer_counts.json> \
        --comparison <build_comparison.json> \
        --measure baseline=<json> --measure candidate=<json> --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPI_STATE_PENSION_TAXPAYERS_2023 = 7_980_000.0
UNITED_KINGDOM = "K02000001"
TWELVE_AREAS = (
    *(f"E1200000{number}" for number in range(1, 10)),
    "W92000004",
    "S92000003",
    "N92000002",
)
WINDOW = {2024: 0.25, 2025: 0.75}
RECIPIENT_TARGETS = (
    "dwp.state_pension.recipients@2025",
    "dfc_ni.state_pension.recipients@2025",
)


def _state_pension_age(rows: list[dict]) -> dict[tuple[str, int], float]:
    return {
        (row["geography"]["id"], int(row["period"]["value"])): float(row["value"])
        for row in rows
        if row.get("measure_id") == "state_pension_age"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendored", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--measure", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    values = _state_pension_age(json.loads(args.vendored.read_text())["rows"])
    share = SPI_STATE_PENSION_TAXPAYERS_2023 / values[(UNITED_KINGDOM, 2023)]
    windows = {
        "twelve_areas": sum(
            weight * values[(area, year)]
            for year, weight in WINDOW.items()
            for area in TWELVE_AREAS
        ),
        "united_kingdom": sum(
            weight * values[(UNITED_KINGDOM, year)] for year, weight in WINDOW.items()
        ),
    }
    comparison = {
        row["name"]: row for row in json.loads(args.comparison.read_text())["rows"]
    }
    recipients = sum(
        comparison[name]["candidate"]["target"] for name in RECIPIENT_TARGETS
    )
    implied = {
        level: {
            "taxpaying": share * window,
            "non_taxpaying": recipients - share * window,
        }
        for level, window in windows.items()
    }
    low, high = sorted(level["taxpaying"] for level in implied.values())

    builds = {}
    for item in args.measure:
        label, _, path = item.partition("=")
        split = json.loads(Path(path).read_text())["state_pension_recipients"][
            "state_pension"
        ]
        builds[label] = {
            "taxpaying": split["taxpaying"],
            "non_taxpaying": split["non_taxpaying"],
            "taxpaying_within_range": low <= split["taxpaying"] <= high,
            "non_taxpaying_within_range": recipients - high
            <= split["non_taxpaying"]
            <= recipients - low,
        }

    output = {
        "spi_state_pension_taxpayers_2023": SPI_STATE_PENSION_TAXPAYERS_2023,
        "united_kingdom_state_pension_age_taxpayers_2023": values[
            (UNITED_KINGDOM, 2023)
        ],
        "share_with_a_state_pension": share,
        "calendar_2025_windows": windows,
        "recipient_target": recipients,
        "implied": implied,
        "builds": builds,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
