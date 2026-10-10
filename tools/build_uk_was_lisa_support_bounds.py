"""Build disclosure-safe WAS Lifetime ISA support bounds from the pinned WAS tabs.

The ``was_lisa`` stage clips its drawn balances to the donor's exact realised
range in-run; the terminal ``uk_support`` gate checks the exported column
against these committed bounds instead, rounded outward to one significant
figure so no committed number is a unit-record value (the rounding of
``tools/build_uk_was_wealth_support_bounds.py``). The donor is cleaned by the
stage's own function under the packaged declaration, so the bounds describe
exactly the credible holders the balance model is fitted on.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.was_lisa import (
    CLEAN_WAS_LISA_DONOR_KIND,
    UK_WAS_HOUSEHOLD_TAB_ROLE,
    UK_WAS_LISA_STAGE_NAME,
    UK_WAS_LISA_SUPPORT_CLIP_COLUMNS,
    UK_WAS_PERSON_TAB_ROLE,
    WASLISAColumns,
    clean_was_lisa_donor,
    was_lisa_support_ranges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "packages/microcosm-build/src/microcosm/build/uk/was_lisa_support_bounds.json"
)


def _stage_declaration() -> tuple[Mapping[str, Mapping[str, Any]], WASLISAColumns]:
    spec = load_country_spec("uk")
    if spec.sources is None:
        raise ValueError("the UK spec declares no source stages.")
    stage = spec.sources.stage_map()[UK_WAS_LISA_STAGE_NAME]
    artifacts = {str(artifact["role"]): artifact for artifact in stage.artifacts}
    parameters = next(
        dict(operation.parameters)
        for operation in stage.operations
        if operation.kind == CLEAN_WAS_LISA_DONOR_KIND
    )
    return artifacts, WASLISAColumns.from_parameters(parameters)


def support_bounds_payload(
    person_raw: pd.DataFrame,
    household_raw: pd.DataFrame,
    *,
    columns: WASLISAColumns,
    household_tab_sha256: str,
    person_tab_sha256: str,
) -> dict[str, object]:
    """The committed bounds payload for already-read WAS tabs."""

    donor = clean_was_lisa_donor(person_raw, household_raw, columns=columns)
    exact = was_lisa_support_ranges(donor.person)
    return {
        "version": 1,
        "country": "uk",
        "policy": (
            "Disclosure-safe outward-rounded WAS Lifetime ISA support bounds for "
            "the UK support gate, generated from the WAS round-8 person and "
            "household tabs recorded under source by "
            "tools/build_uk_was_lisa_support_bounds.py. Values are the "
            "donor-realized exact ranges of the credible holders (after the "
            "stage's declared credibility rule) and the zero balances of "
            "non-holders, rounded outward to one significant figure, so "
            "unit-record minima and maxima are never committed."
        ),
        "source": {
            "ukds_study_number": 7215,
            "doi": "10.5255/UKDA-SN-7215-20",
            "household_artifact": "was_round_8_hhold_eul_may_2025_230525.tab",
            "household_tab_sha256": household_tab_sha256,
            "person_artifact": "was_round_8_person_eul_may_2025_230525.tab",
            "person_tab_sha256": person_tab_sha256,
            "sdc_treatment": (
                "Exact donor min/max values are rounded outward to one "
                "significant figure before commit, the treatment adjudicated "
                "for the WAS wealth bounds by juaristi22 on 2026-08-19 "
                "(microcosm#714)."
            ),
        },
        "bounds": {
            column: list(_outward_round(bounds))
            for column, bounds in exact.items()
            if column in UK_WAS_LISA_SUPPORT_CLIP_COLUMNS
        },
        "chronicle": [
            "Support bounds generated from the tabs pinned as "
            f"source.household_tab_sha256 ({household_tab_sha256[:12]}…) and "
            f"source.person_tab_sha256 ({person_tab_sha256[:12]}…) with outward "
            "SDC rounding (microcosm#1003)."
        ],
    }


def build_support_bounds(was_tab: Path, was_person_tab: Path) -> dict[str, object]:
    """Read both pinned tabs (size and sha256 checked) and build the payload."""

    artifacts, columns = _stage_declaration()
    household_artifact = artifacts[UK_WAS_HOUSEHOLD_TAB_ROLE]
    person_artifact = artifacts[UK_WAS_PERSON_TAB_ROLE]
    household = read_pinned_tab(was_tab, household_artifact)
    person = read_pinned_tab(
        was_person_tab, person_artifact, columns=columns.person_raw_columns()
    )
    return support_bounds_payload(
        person,
        household,
        columns=columns,
        household_tab_sha256=str(household_artifact["sha256"]),
        person_tab_sha256=str(person_artifact["sha256"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--was-tab", type=Path, required=True)
    parser.add_argument("--was-person-tab", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build_support_bounds(args.was_tab, args.was_person_tab)
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
