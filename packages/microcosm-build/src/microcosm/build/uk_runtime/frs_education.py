"""FRS education derivations for the UK source spine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import (
    WEEKS_IN_YEAR,
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

NON_ADVANCED_EDUCATION_LEVELS = (
    "PRE_PRIMARY",
    "PRIMARY",
    "LOWER_SECONDARY",
    "UPPER_SECONDARY",
    "POST_SECONDARY",
)
FRS_APPROVED_TRAINING_CODES = tuple(range(1, 10))
UNKNOWN_QUALIFYING_EDUCATION_OR_TRAINING_ENTRY_AGE = 1000
BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS = (
    "universal_credit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)
EDUCQUAL_MAP = {
    1: "NOT_COMPLETED_PRIMARY",
    2: "LOWER_SECONDARY",
    3: "LOWER_SECONDARY",
    4: "UPPER_SECONDARY",
    5: "UPPER_SECONDARY",
    6: "UPPER_SECONDARY",
    7: "UPPER_SECONDARY",
    8: "LOWER_SECONDARY",
    9: "UPPER_SECONDARY",
    10: "UPPER_SECONDARY",
    11: "POST_SECONDARY",
    12: "POST_SECONDARY",
    13: "UPPER_SECONDARY",
    14: "POST_SECONDARY",
    15: "UPPER_SECONDARY",
    16: "POST_SECONDARY",
    17: "TERTIARY",
    18: "TERTIARY",
    19: "TERTIARY",
    20: "TERTIARY",
    21: "TERTIARY",
    66: "UPPER_SECONDARY",
    67: "UPPER_SECONDARY",
    68: "UPPER_SECONDARY",
    69: "POST_SECONDARY",
    70: "TERTIARY",
    **{code: "POST_SECONDARY" for code in range(22, 66)},
    **{code: "POST_SECONDARY" for code in range(71, 86)},
    86: "UPPER_SECONDARY",
    87: "UPPER_SECONDARY",
}
FRS_EDUCATION_OUTPUT_COLUMNS = (
    "current_education",
    "highest_education",
    "is_in_non_advanced_education",
    "is_in_approved_training",
    "age_started_or_accepted_current_education_or_training",
    "is_before_universal_credit_qualifying_young_person_terminal_date",
    "adult_ema",
    "child_ema",
    "receives_benefits_in_own_right",
)


class UKFRSEducationStageTransform:
    """Whole-stage callable for FRS education derivations."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_education(frame, self.raw_dir, stage=self.stage)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_EDUCATION_OUTPUT_COLUMNS


def add_frs_education(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> Frame:
    artifacts = _artifact_by_table(stage)
    adult = normalize_ids(
        read_pinned_tab(Path(raw_dir) / str(artifacts["adult"]["locator"]), artifacts["adult"])
    )
    child = normalize_ids(
        read_pinned_tab(Path(raw_dir) / str(artifacts["child"]["locator"]), artifacts["child"])
    )
    raw_person = pd.concat([adult, child], ignore_index=True, sort=False)
    person = frame.table("person").copy()
    derived = derive_frs_education(person, raw_person)
    for column in FRS_EDUCATION_OUTPUT_COLUMNS:
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


def derive_frs_education(person: pd.DataFrame, raw_person: pd.DataFrame) -> pd.DataFrame:
    raw = raw_person.set_index("person_id").reindex(person["person_id"])
    values = pd.DataFrame(index=person.index)
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0).to_numpy()
    fted_source = "fted" if "fted" in raw.columns else "educft"
    fted = _num(raw, fted_source).to_numpy()
    typeed2 = _num(raw, "typeed2").to_numpy()
    current = derive_current_education(fted=fted, typeed2=typeed2, age=age)
    values["current_education"] = current
    values["highest_education"] = (
        _num(raw, "educqual").astype(int).map(EDUCQUAL_MAP).fillna("UPPER_SECONDARY")
    ).to_numpy()
    values["is_in_non_advanced_education"] = np.isin(
        current, NON_ADVANCED_EDUCATION_LEVELS
    )
    train = _num(raw, "train") if "train" in raw.columns else pd.Series(0, index=raw.index)
    values["is_in_approved_training"] = train.isin(FRS_APPROVED_TRAINING_CODES).to_numpy()
    in_qualifying = (
        values["is_in_non_advanced_education"].to_numpy(dtype=bool)
        | values["is_in_approved_training"].to_numpy(dtype=bool)
    )
    values["age_started_or_accepted_current_education_or_training"] = np.where(
        in_qualifying,
        np.minimum(age, 18),
        UNKNOWN_QUALIFYING_EDUCATION_OR_TRAINING_ENTRY_AGE,
    )
    values["is_before_universal_credit_qualifying_young_person_terminal_date"] = (
        (age == 19) & in_qualifying
    )
    values["adult_ema"] = _ema(raw, amount_column=("emaamt", "edumaamt"))
    values["child_ema"] = _ema(raw, amount_column=("chemaamt", "eduma"))
    benefit_columns = [
        column for column in BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS if column in person
    ]
    values["receives_benefits_in_own_right"] = (
        person[benefit_columns].fillna(0).sum(axis=1) > 0 if benefit_columns else False
    )
    return values


def derive_current_education(*, fted, typeed2, age) -> np.ndarray:
    fted = np.asarray(fted)
    typeed2 = np.asarray(typeed2)
    age = np.asarray(age)
    # The post-secondary branch is deliberately unreachable because the
    # incumbent tests upper-secondary first for typeed2 7/8.
    return np.select(
        [
            np.isin(fted, (2, -1, 0)),
            typeed2 == 1,
            np.isin(typeed2, (2, 4))
            | (np.isin(typeed2, (3, 8)) & (age < 11))
            | ((typeed2 == 0) & (fted == 1) & (age > 5) & (age < 11)),
            np.isin(typeed2, (5, 6))
            | (np.isin(typeed2, (3, 8)) & (age >= 11) & (age <= 16))
            | ((typeed2 == 0) & (fted == 1) & (age <= 16)),
            (typeed2 == 7)
            | (np.isin(typeed2, (3, 8)) & (age > 16))
            | ((typeed2 == 0) & (fted == 1) & (age > 16)),
            np.isin(typeed2, (7, 8)) & (age >= 19),
            (typeed2 == 9) | ((typeed2 == 0) & (fted == 1) & (age >= 19)),
        ],
        [
            "NOT_IN_EDUCATION",
            "PRE_PRIMARY",
            "PRIMARY",
            "LOWER_SECONDARY",
            "UPPER_SECONDARY",
            "POST_SECONDARY",
            "TERTIARY",
        ],
        default="NOT_IN_EDUCATION",
    )


def _ema(raw: pd.DataFrame, *, amount_column: tuple[str, str]) -> np.ndarray:
    for column in amount_column:
        if column in raw.columns:
            amount = _num(raw, column)
            return np.maximum(amount, 0).to_numpy(dtype=float) * WEEKS_IN_YEAR
    return np.zeros(len(raw), dtype=float)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    missing = sorted({"adult", "child"} - set(by_table))
    if missing:
        raise ValueError(f"frs_education manifest is missing tab artifact(s): {missing}.")
    return by_table
