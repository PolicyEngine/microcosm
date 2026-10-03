"""FRS derived employment columns for the UK source spine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
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

# Adult-table EMPSTATI ("Adult - Employment Status - ILO definition") value
# labels from the UKDS FRS 2024-25 data dictionary (SN 9563), mapped to
# policyengine-uk EmploymentStatus names. Matches the incumbent's
# FRS_EMPSTATI_EMPLOYMENT_STATUS (uk-data#526). Child-table people have no
# EMPSTATI and are CHILD.
FRS_EMPSTATI_EMPLOYMENT_STATUS = MappingProxyType(
    {
        1: "FT_EMPLOYED",  # Full-time employee
        2: "PT_EMPLOYED",  # Part-time employee
        3: "FT_SELF_EMPLOYED",  # Full-time self-employed
        4: "PT_SELF_EMPLOYED",  # Part-time self-employed
        5: "UNEMPLOYED",  # Unemployed
        6: "RETIRED",  # Retired
        7: "STUDENT",  # Student
        8: "CARER",  # Looking after family/home
        9: "LONG_TERM_DISABLED",  # Permanently sick/disabled
        10: "SHORT_TERM_DISABLED",  # Temporarily sick/injured
        11: "OTHER_INACTIVE",  # Other inactive
    }
)
CHILD_EMPLOYMENT_STATUS = "CHILD"
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
        read_pinned_tab(
            Path(raw_dir) / str(artifacts["adult"]["locator"]), artifacts["adult"]
        )
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
    A person is an adult record when their ``person_id`` is in ``adult.tab``;
    everyone else is CHILD.
    """

    raw = adult.set_index("person_id")
    aligned = raw.reindex(person["person_id"])
    values = pd.DataFrame(index=person.index)
    values["employment_status"] = derive_employment_status_from_frs(
        aligned["empstati"], person["person_id"].isin(adult["person_id"])
    )
    sector = pd.to_numeric(aligned["mjobsect"], errors="coerce").fillna(0)
    values["employment_sector"] = (
        sector.astype(int).map(EMPLOYMENT_SECTOR_MAP).fillna("NOT_EMPLOYED").to_numpy()
    )
    values["sic_industry_division"] = (
        pd.to_numeric(aligned["sic"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(int)
        .to_numpy()
    )
    return values


def derive_employment_status_from_frs(empstati, is_adult_record) -> np.ndarray:
    """Map FRS EMPSTATI codes to ``employment_status``.

    People who are not adult records (child-table people) are CHILD whatever
    their code. Every adult must carry a code in
    ``FRS_EMPSTATI_EMPLOYMENT_STATUS``: a missing or unknown adult code refuses
    the build rather than falling back to a guessed status, as the old fallback
    silently made code 11 (Other inactive) LONG_TERM_DISABLED. The refusal
    names the offending codes but never how many adults carry them.
    """

    codes = pd.Series(
        np.asarray(pd.to_numeric(empstati, errors="coerce"), dtype="float64")
    )
    adult = np.asarray(is_adult_record, dtype=bool)
    if adult.shape != codes.shape:
        raise ValueError(
            "FRS EMPSTATI codes and adult-record flags must have the same length."
        )
    adult_status = codes.map(FRS_EMPSTATI_EMPLOYMENT_STATUS).to_numpy(dtype=object)
    unknown = adult & pd.isna(adult_status)
    if unknown.any():
        bad = codes[unknown]
        labels = [_code_label(code) for code in sorted(bad.dropna().unique())]
        if bad.isna().any():
            labels.append("blank or non-numeric")
        # Build logs are not licensed outputs, and adults are not survey
        # households, so the refusal names the codes and never a count.
        raise ValueError(
            "FRS adult.tab carries adults with EMPSTATI code(s) "
            f"{labels} outside FRS_EMPSTATI_EMPLOYMENT_STATUS; map them from the "
            "release's data dictionary."
        )
    return np.where(adult, adult_status, CHILD_EMPLOYMENT_STATUS).astype(object)


def _code_label(code: float) -> str:
    return str(int(code)) if float(code).is_integer() else str(code)


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    if "adult" not in by_table:
        raise ValueError("frs_employment manifest is missing adult.tab.")
    return by_table
