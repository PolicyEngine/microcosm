"""FRS derived employment columns for the UK source spine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import (
    normalize_ids,
    read_pinned_tab,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame

EMPLOYMENT_STATUS_MAP = {
    0: "CHILD",
    1: "FT_EMPLOYED",
    2: "PT_EMPLOYED",
    3: "FT_SELF_EMPLOYED",
    4: "PT_SELF_EMPLOYED",
    5: "UNEMPLOYED",
    6: "RETIRED",
    7: "STUDENT",
    8: "CARER",
    9: "LONG_TERM_DISABLED",
    10: "SHORT_TERM_DISABLED",
    11: "LONG_TERM_DISABLED",
}
EMPLOYMENT_SECTOR_MAP = {
    0: "NOT_EMPLOYED",
    1: "PRIVATE",
    2: "PUBLIC",
}
FRS_EMPLOYMENT_OUTPUT_COLUMNS = (
    "employment_status",
    "employment_sector",
    "sic_industry_division",
)


class UKFRSEmploymentStageTransform:
    """Whole-stage callable for FRS employment derivations."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_employment(frame, self.raw_dir, stage=self.stage)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_EMPLOYMENT_OUTPUT_COLUMNS


def add_frs_employment(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> Frame:
    """Add employment status, sector, and SIC from pinned ``adult.tab``."""

    artifacts = _artifact_by_table(stage)
    adult = normalize_ids(
        read_pinned_tab(Path(raw_dir) / str(artifacts["adult"]["locator"]), artifacts["adult"])
    )
    derived = derive_frs_employment(frame.table("person"), adult)
    person = frame.table("person").copy()
    for column in FRS_EMPLOYMENT_OUTPUT_COLUMNS:
        person[column] = derived[column].to_numpy()
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def derive_frs_employment(person: pd.DataFrame, adult: pd.DataFrame) -> pd.DataFrame:
    """Return person-indexed employment derivations.

    ``empstati``, ``mjobsect``, and ``sic`` are intentionally direct-indexed
    so missing source columns fail loudly, matching the incumbent FRS port.
    """

    raw = adult.set_index("person_id")
    aligned = raw.reindex(person["person_id"])
    values = pd.DataFrame(index=person.index)
    empstati = pd.to_numeric(aligned["empstati"], errors="coerce").fillna(0)
    # Unmapped codes above the declared domain follow the incumbent's
    # post-map fillna to LONG_TERM_DISABLED; 0/NaN rows land on CHILD via
    # the explicit map entry.
    values["employment_status"] = (
        empstati.astype(int)
        .map(EMPLOYMENT_STATUS_MAP)
        .fillna("LONG_TERM_DISABLED")
        .to_numpy()
    )
    sector = pd.to_numeric(aligned["mjobsect"], errors="coerce").fillna(0)
    values["employment_sector"] = (
        sector.astype(int).map(EMPLOYMENT_SECTOR_MAP).fillna("NOT_EMPLOYED").to_numpy()
    )
    values["sic_industry_division"] = (
        pd.to_numeric(aligned["sic"], errors="coerce").fillna(0).clip(lower=0).astype(int).to_numpy()
    )
    return values


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    if "adult" not in by_table:
        raise ValueError("frs_employment manifest is missing adult.tab.")
    return by_table
