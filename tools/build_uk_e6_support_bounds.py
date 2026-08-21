"""Build disclosure-safe E6 UK support bounds from pinned donor tabs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.etb_services import clean_etb_services_table
from microcosm.build.uk_runtime.etb_services import (
    donor_realized_ranges as etb_services_ranges,
)
from microcosm.build.uk_runtime.etb_vat import clean_etb_vat_table
from microcosm.build.uk_runtime.etb_vat import (
    donor_realized_ranges as etb_vat_ranges,
)
from microcosm.build.uk_runtime.lcfs_consumption import clean_lcfs_consumption_table
from microcosm.build.uk_runtime.lcfs_consumption import (
    donor_realized_ranges as lcfs_ranges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"


def build_lcfs_support_bounds(
    household_tab: Path, person_tab: Path
) -> dict[str, object]:
    hh_sha = _sha256(household_tab)
    person_sha = _sha256(person_tab)
    donor = clean_lcfs_consumption_table(
        pd.read_csv(person_tab, sep="\t", low_memory=False),
        pd.read_csv(household_tab, sep="\t", low_memory=False),
    )
    return _payload(
        source={
            "ukds_study_number": 9468,
            "doi": "10.5255/UKDA-SN-9468-3",
            "household_tab_sha256": hh_sha,
            "person_tab_sha256": person_sha,
        },
        bounds=lcfs_ranges(donor),
        label="LCFS consumption",
    )


def build_etb_vat_support_bounds(etb_tab: Path) -> dict[str, object]:
    sha = _sha256(etb_tab)
    donor = clean_etb_vat_table(pd.read_csv(etb_tab, sep="\t", low_memory=False))
    return _payload(
        source={
            "ukds_study_number": 8856,
            "doi": "10.5255/UKDA-SN-8856-4",
            "tab_sha256": sha,
        },
        bounds=etb_vat_ranges(donor),
        label="ETB VAT",
    )


def build_etb_services_support_bounds(etb_tab: Path) -> dict[str, object]:
    sha = _sha256(etb_tab)
    donor = clean_etb_services_table(pd.read_csv(etb_tab, sep="\t", low_memory=False))
    return _payload(
        source={
            "ukds_study_number": 8856,
            "doi": "10.5255/UKDA-SN-8856-4",
            "tab_sha256": sha,
        },
        bounds=etb_services_ranges(donor),
        label="ETB services",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcfs-hh-tab", type=Path)
    parser.add_argument("--lcfs-person-tab", type=Path)
    parser.add_argument("--etb-tab", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    jobs: list[tuple[Path, dict[str, object]]] = []
    if args.lcfs_hh_tab or args.lcfs_person_tab:
        if not args.lcfs_hh_tab or not args.lcfs_person_tab:
            raise SystemExit("LCFS support bounds require both LCFS tabs.")
        jobs.append(
            (
                UK_PACKAGE / "lcfs_consumption_support_bounds.json",
                build_lcfs_support_bounds(args.lcfs_hh_tab, args.lcfs_person_tab),
            )
        )
    if args.etb_tab:
        jobs.extend(
            [
                (
                    UK_PACKAGE / "etb_vat_support_bounds.json",
                    build_etb_vat_support_bounds(args.etb_tab),
                ),
                (
                    UK_PACKAGE / "etb_services_support_bounds.json",
                    build_etb_services_support_bounds(args.etb_tab),
                ),
            ]
        )
    if not jobs:
        raise SystemExit("No support-bound inputs supplied.")
    for path, payload in jobs:
        rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        if args.check:
            if path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"{path} is stale.")
        else:
            path.write_text(rendered, encoding="utf-8")
    return 0


def _payload(
    *,
    source: dict[str, object],
    bounds: dict[str, tuple[float, float]],
    label: str,
) -> dict[str, object]:
    return {
        "version": 1,
        "country": "uk",
        "policy": (
            f"Disclosure-safe outward-rounded {label} support bounds generated "
            "from pinned licensed donor tabs. Values are rounded outward to "
            "one significant figure; exact donor min/max values are not committed."
        ),
        "source": {
            **source,
            "sdc_treatment": (
                "Exact donor min/max values are rounded outward to one "
                "significant figure before commit."
            ),
        },
        "bounds": {
            column: list(_outward_round(pair))
            for column, pair in sorted(bounds.items())
        },
        "chronicle": [f"Support bounds generated for {label} from source SHA pins."],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
