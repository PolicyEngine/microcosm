"""FRS disability benefit category and flag derivations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame

UK_INTERNAL_DISABILITY_REPORTED_COLUMNS = (
    "attendance_allowance_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
)
UK_DISABILITY_FLAG_REPORTED_COLUMNS = (
    *UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
    "sda_reported",
    "incapacity_benefit_reported",
    "iidb_reported",
    "afcs_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)
FRS_DISABILITY_OUTPUT_COLUMNS = (
    "aa_category",
    "dla_sc_category",
    "dla_m_category",
    "pip_m_category",
    "pip_dl_category",
    "is_disabled_for_benefits",
    "is_enhanced_disabled_for_benefits",
    "is_severely_disabled_for_benefits",
)


@dataclass(frozen=True)
class UKDWPBaselineDisabilityRates:
    """Baseline DWP weekly rates read at 1 January of the build year."""

    aa_lower: float
    aa_higher: float
    dla_sc_lower: float
    dla_sc_middle: float
    dla_sc_higher: float
    dla_m_lower: float
    dla_m_higher: float
    pip_m_standard: float
    pip_m_enhanced: float
    pip_dl_standard: float
    pip_dl_enhanced: float
    instant: str
    source: str


@dataclass(frozen=True)
class UKDWPDisabilityFlagRates:
    """Plain DWP weekly flag rates read at 1 January of the build year."""

    aa_higher: float
    dla_sc_higher: float
    pip_dl_enhanced: float
    instant: str
    source: str


class UKFRSDisabilityStageTransform:
    """Whole-stage callable for FRS disability derivations."""

    def __init__(
        self,
        *,
        stage: SourceStageSpec,
        baseline_rates: UKDWPBaselineDisabilityRates | None = None,
        flag_rates: UKDWPDisabilityFlagRates | None = None,
    ) -> None:
        self.stage = stage
        self.baseline_rates = baseline_rates
        self.flag_rates = flag_rates

    def __call__(self, frame: Frame) -> Frame:
        period = uk_time_period(frame)
        return add_frs_disability(
            frame,
            baseline_rates=self.baseline_rates
            or uk_dwp_baseline_disability_rates(period),
            flag_rates=self.flag_rates or uk_dwp_disability_flag_rates(period),
        )

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_DISABILITY_OUTPUT_COLUMNS


def uk_dwp_baseline_disability_rates(
    build_period: int | str,
) -> UKDWPBaselineDisabilityRates:
    """Read baseline DWP disability rates at ``{year}-01-01``.

    This intentionally differs from the CGT reader's June instant: the
    incumbent's ``parameters(year)`` resolves at January 1, before April
    DWP uprating.
    """

    try:
        import policyengine_uk
    except ImportError as exc:
        raise ImportError(
            "UK DWP disability parameters require `uv sync --all-packages --extra uk`."
        ) from exc

    instant = f"{int(build_period)}-01-01"
    dwp = (
        policyengine_uk.CountryTaxBenefitSystem()
        .parameters(int(build_period))
        .baseline.gov.dwp
    )
    return UKDWPBaselineDisabilityRates(
        aa_lower=_parameter_value(dwp.attendance_allowance.lower, instant),
        aa_higher=_parameter_value(dwp.attendance_allowance.higher, instant),
        dla_sc_lower=_parameter_value(dwp.dla.self_care.lower, instant),
        dla_sc_middle=_parameter_value(dwp.dla.self_care.middle, instant),
        dla_sc_higher=_parameter_value(dwp.dla.self_care.higher, instant),
        dla_m_lower=_parameter_value(dwp.dla.mobility.lower, instant),
        dla_m_higher=_parameter_value(dwp.dla.mobility.higher, instant),
        pip_m_standard=_parameter_value(dwp.pip.mobility.standard, instant),
        pip_m_enhanced=_parameter_value(dwp.pip.mobility.enhanced, instant),
        pip_dl_standard=_parameter_value(dwp.pip.daily_living.standard, instant),
        pip_dl_enhanced=_parameter_value(dwp.pip.daily_living.enhanced, instant),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def uk_dwp_disability_flag_rates(
    build_period: int | str,
) -> UKDWPDisabilityFlagRates:
    """Read plain DWP disability flag rates at ``{year}-01-01``."""

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "UK DWP disability parameters require `uv sync --all-packages --extra uk`."
        ) from exc

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    instant = f"{int(build_period)}-01-01"
    dwp = parameters.gov.dwp
    return UKDWPDisabilityFlagRates(
        aa_higher=float(dwp.attendance_allowance.higher(instant)),
        dla_sc_higher=float(dwp.dla.self_care.higher(instant)),
        pip_dl_enhanced=float(dwp.pip.daily_living.enhanced(instant)),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def add_frs_disability(
    frame: Frame,
    *,
    baseline_rates: UKDWPBaselineDisabilityRates,
    flag_rates: UKDWPDisabilityFlagRates,
) -> Frame:
    person = frame.table("person").copy()
    derived = derive_frs_disability(
        person, baseline_rates=baseline_rates, flag_rates=flag_rates
    )
    for column in FRS_DISABILITY_OUTPUT_COLUMNS:
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


def derive_frs_disability(
    person: pd.DataFrame,
    *,
    baseline_rates: UKDWPBaselineDisabilityRates,
    flag_rates: UKDWPDisabilityFlagRates,
) -> pd.DataFrame:
    values = pd.DataFrame(index=person.index)
    values["aa_category"] = _category(
        _amount(person, "attendance_allowance_reported"),
        (("LOWER", baseline_rates.aa_lower), ("HIGHER", baseline_rates.aa_higher)),
    )
    values["dla_sc_category"] = _category(
        _amount(person, "dla_sc_reported"),
        (
            ("LOWER", baseline_rates.dla_sc_lower),
            ("MIDDLE", baseline_rates.dla_sc_middle),
            ("HIGHER", baseline_rates.dla_sc_higher),
        ),
    )
    values["dla_m_category"] = _category(
        _amount(person, "dla_m_reported"),
        (("LOWER", baseline_rates.dla_m_lower), ("HIGHER", baseline_rates.dla_m_higher)),
    )
    values["pip_m_category"] = _category(
        _amount(person, "pip_m_reported"),
        (
            ("STANDARD", baseline_rates.pip_m_standard),
            ("ENHANCED", baseline_rates.pip_m_enhanced),
        ),
    )
    values["pip_dl_category"] = _category(
        _amount(person, "pip_dl_reported"),
        (
            ("STANDARD", baseline_rates.pip_dl_standard),
            ("ENHANCED", baseline_rates.pip_dl_enhanced),
        ),
    )
    total = sum(_amount(person, column) for column in UK_DISABILITY_FLAG_REPORTED_COLUMNS)
    dla_sc = _amount(person, "dla_sc_reported")
    aa = _amount(person, "attendance_allowance_reported")
    pip_dl = _amount(person, "pip_dl_reported")
    afcs = _amount(person, "afcs_reported")
    gap = WEEKS_IN_YEAR
    aa_higher = flag_rates.aa_higher * WEEKS_IN_YEAR - gap
    dla_sc_higher = flag_rates.dla_sc_higher * WEEKS_IN_YEAR - gap
    pip_dl_enhanced = flag_rates.pip_dl_enhanced * WEEKS_IN_YEAR - gap
    values["is_disabled_for_benefits"] = total > 0
    values["is_enhanced_disabled_for_benefits"] = (
        (aa >= aa_higher) | (dla_sc > dla_sc_higher) | (pip_dl >= pip_dl_enhanced)
    )
    values["is_severely_disabled_for_benefits"] = (
        (aa > 0) | (dla_sc >= dla_sc_higher) | (pip_dl >= pip_dl_enhanced) | (afcs > 0)
    )
    return values


def _amount(person: pd.DataFrame, column: str) -> pd.Series:
    if column not in person:
        return pd.Series(0.0, index=person.index)
    return pd.to_numeric(person[column], errors="coerce").fillna(0.0)


def _category(
    reported_amount: pd.Series, thresholds: tuple[tuple[str, float], ...]
) -> np.ndarray:
    weekly = reported_amount.to_numpy(dtype=float) / WEEKS_IN_YEAR
    category = np.full(len(weekly), "NONE", dtype=object)
    for name, weekly_rate in thresholds:
        category[weekly >= max(0.0, float(weekly_rate) - 1.0)] = name
    return category


def _parameter_value(value, instant: str) -> float:
    if callable(value):
        value = value(instant)
    return float(value)
