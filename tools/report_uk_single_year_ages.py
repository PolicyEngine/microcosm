#!/usr/bin/env python
"""Report a built UK file's single years of age against ONS's mid-2025 estimates.

uk-data#600's age check (microcosm#1095): the file's weighted persons by sex
and single year of age (0 to 89, and 90 and over) beside ONS MYE2 mid-2025,
with their ratio, and the tail bands 80 to 84, 85 to 89, 90 and over and 80
and over. It reads a built H5's frame tables and final household weights (no
engine). Report-only: the estimates bind nothing. Every cell names its
carriers as distinct source households and reads "<3" in place of its
figures when fewer than 3 carry it; when any single year of a sex is
withheld, every cell of that sex is (UKDS disclosure control). The receipt
belongs in the run tree.

    uv run python tools/report_uk_single_year_ages.py \\
        --h5 <run>/microcosm_uk_2024_25.h5 --output <run>/single_year_ages.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
    from microcosm.build.uk_runtime.single_year_ages import single_year_age_report

    frame, _ = load_uk_national_frame(args.h5)
    receipt = single_year_age_report(
        frame.table("person"),
        frame.table("household"),
        frame.weights_for("household").values,
    )
    receipt["h5"] = str(args.h5)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
