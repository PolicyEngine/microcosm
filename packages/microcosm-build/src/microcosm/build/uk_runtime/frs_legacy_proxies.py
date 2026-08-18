"""FRS legacy-benefit claimant-state proxies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from microcosm.frame.rules import assert_rules_engine_country

FRS_LEGACY_PROXY_OUTPUT_COLUMNS = (
    "legacy_jobseeker_proxy",
    "esa_health_condition_proxy",
    "esa_support_group_proxy",
)
UK_LEGACY_PROXY_PREDICTORS = ("state_pension_age",)
ESA_HEALTH_EMPLOYMENT_STATUSES = ("LONG_TERM_DISABLED", "SHORT_TERM_DISABLED")


@dataclass(frozen=True)
class UKLegacyJSAPolicy:
    """JSA hours rule read at 1 January of the build year."""

    max_weekly_hours_single: float
    instant: str
    source: str


class UKFRSLegacyProxiesStageTransform:
    """Whole-stage callable for FRS legacy-benefit proxies."""

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        stage: SourceStageSpec,
        engine: object,
        policy: UKLegacyJSAPolicy | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage
        self.engine = engine
        self.policy = policy

    def __call__(self, frame: Frame) -> Frame:
        period = uk_time_period(frame)
        assert_rules_engine_country(self.engine, "uk")
        materialized = self.engine.materialize(frame, UK_LEGACY_PROXY_PREDICTORS, period)
        return add_frs_legacy_proxies(
            frame,
            self.raw_dir,
            stage=self.stage,
            state_pension_age=materialized["state_pension_age"],
            policy=self.policy or uk_legacy_jsa_policy(period),
        )

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_LEGACY_PROXY_OUTPUT_COLUMNS


def uk_legacy_jsa_policy(build_period: int | str) -> UKLegacyJSAPolicy:
    """Read JSA weekly hours at ``{year}-01-01``."""

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "UK JSA parameters require `uv sync --all-packages --extra uk`."
        ) from exc

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    instant = f"{int(build_period)}-01-01"
    return UKLegacyJSAPolicy(
        max_weekly_hours_single=float(parameters.gov.dwp.JSA.hours.single(instant)),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def add_frs_legacy_proxies(
    frame: Frame,
    raw_dir: str | Path,
    *,
    stage: SourceStageSpec,
    state_pension_age: np.ndarray,
    policy: UKLegacyJSAPolicy,
) -> Frame:
    artifacts = _artifact_by_table(stage)
    adult = normalize_ids(
        read_pinned_tab(Path(raw_dir) / str(artifacts["adult"]["locator"]), artifacts["adult"])
    )
    person = frame.table("person").copy()
    raw = adult.set_index("person_id").reindex(person["person_id"])
    reported = pd.to_numeric(raw["empstati"], errors="coerce").fillna(0).to_numpy() > 0
    derived = derive_frs_legacy_proxies(
        person,
        employment_status_reported=reported,
        state_pension_age=state_pension_age,
        max_annual_hours=policy.max_weekly_hours_single * WEEKS_IN_YEAR,
    )
    for column in FRS_LEGACY_PROXY_OUTPUT_COLUMNS:
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


def derive_frs_legacy_proxies(
    person: pd.DataFrame,
    *,
    employment_status_reported,
    state_pension_age,
    max_annual_hours: float,
) -> pd.DataFrame:
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0).to_numpy()
    status = person["employment_status"].to_numpy()
    hours = pd.to_numeric(person["hours_worked"], errors="coerce").fillna(0).to_numpy()
    education = person["current_education"].to_numpy()
    reported = np.asarray(employment_status_reported, dtype=bool)
    spa = np.asarray(state_pension_age, dtype=float)
    values = pd.DataFrame(index=person.index)
    values["legacy_jobseeker_proxy"] = (
        reported
        & (age >= 18)
        & (age < spa)
        & (status == "UNEMPLOYED")
        & (hours < max_annual_hours)
        & (education == "NOT_IN_EDUCATION")
    )
    health = reported & (age >= 16) & (age < spa) & np.isin(
        status, ESA_HEALTH_EMPLOYMENT_STATUSES
    )
    values["esa_health_condition_proxy"] = health
    values["esa_support_group_proxy"] = (
        reported & health & (status == "LONG_TERM_DISABLED") & (hours <= 0)
    )
    return values


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    if "adult" not in by_table:
        raise ValueError("frs_legacy_proxies manifest is missing adult.tab.")
    return by_table
