"""FRS disability benefit category and flag derivations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.carers_allowance_take_up import (
    lineage_balanced_positions,
)
from microcosm.build.uk_runtime.cgt_structure import (
    _assert_closed_world_operations,
)
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import (
    FRS_CARE_OUTSIDE_BIT,
    FRS_CARE_RECIPIENTS_COLUMN,
    WEEKS_IN_YEAR,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    UK_WITHHELD,
    rounded_count,
    small_cell_share,
    support_band,
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
YEAR_RULE = "survey_year"
#: The carer link (A3 of the November event-data plan's PR3, microcosm#1095;
#: María's choices of 2026-11-10): policyengine-uk's input
#: ``cares_for_severely_disabled_person`` (2.177.3, pe-uk#2860), whether the
#: person a carer cares for gets a benefit that makes them severely disabled for
#: Carer's Allowance or a qualifying disability benefit for Carer Support
#: Payment. The engine reads it only where care_hours reach the minimum. It is
#: true for a carer whose main care recipients (frs_spine.frs_care_recipients)
#: include a household member with Attendance Allowance, DLA care (Child
#: Disability Payment's included) at the middle or highest rate, or PIP or Adult
#: Disability Payment daily living. A carer whose main recipients include
#: someone outside the household and no such member is drawn at the share, by
#: design weight, of the in-household carers whose care hours reach the minimum
#: and who report no Carer's Allowance that have such a member, on an
#: identity-keyed uniform balanced by weight within the strata of care hours at
#: or above the minimum and below it, as carers_allowance_take_up drew its pool
#: (María's decision (a) of 2026-10-28). A carer with care hours and no link in
#: the care tab keeps the engine's default, true; everyone else is false. Armed
#: Forces Independence Payment is not seen: the FRS can't tell it from other
#: Armed Forces Compensation Scheme payments. The SPI channel takes the link
#: again on its households from the benefits it re-imputes (spi_income).
CARER_LINK_COLUMN = "cares_for_severely_disabled_person"
CARER_LINK_PARAMETERS = "the carer link from frs_care_recipients and the disability categories, with gov.dwp.carers_allowance.min_hours at the survey year"
CARER_LINK_SEED = 0
CARER_LINK_SALT = "cares_for_severely_disabled_person_outside_household"
#: The categories of a qualifying benefit for the carer link.
CARER_LINK_QUALIFYING = {
    "aa_category": ("LOWER", "HIGHER"),
    "dla_sc_category": ("MIDDLE", "HIGHER"),
    "pip_dl_category": ("STANDARD", "ENHANCED"),
}


@dataclass(frozen=True)
class UKDWPDisabilityCategoryRates:
    """Category thresholds from the fiscal-converted ``gov.dwp`` tree."""

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
    """Flag thresholds from the fiscal-converted ``gov.dwp`` tree."""

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
        category_rates: UKDWPDisabilityCategoryRates | None = None,
        flag_rates: UKDWPDisabilityFlagRates | None = None,
        carer_min_hours: float | None = None,
    ) -> None:
        self.stage = stage
        self.category_rates = category_rates
        self.flag_rates = flag_rates
        self.carer_min_hours = carer_min_hours
        self._carer_link: dict[str, object] | None = None

    def __call__(self, frame: Frame) -> Frame:
        _assert_frs_disability_stage_parameters(self.stage)
        year = resolve_uk_year_rule(YEAR_RULE)
        evidence: dict[str, object] = {}
        result = add_frs_disability(
            frame,
            category_rates=self.category_rates
            or uk_dwp_disability_category_rates(year),
            flag_rates=self.flag_rates or uk_dwp_disability_flag_rates(year),
            carer_min_hours=(
                self.carer_min_hours
                if self.carer_min_hours is not None
                else uk_dwp_carers_allowance_min_hours(year)
            ),
            evidence=evidence,
        )
        self._carer_link = dict(evidence["carer_link"])
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (*FRS_DISABILITY_OUTPUT_COLUMNS, CARER_LINK_COLUMN)

    def checkpoint_metadata(self) -> dict[str, object]:
        """Report how the carer link fell, by route, in rounded counts."""

        if self._carer_link is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "evidence": {
                "stage": "frs_disability",
                "carer_link": dict(self._carer_link),
            }
        }


def uk_dwp_disability_category_rates(
    year: int,
) -> UKDWPDisabilityCategoryRates:
    """Read category rates from fiscal-converted ``gov.dwp`` at ``year``.

    The parameter instant is the fiscal-year label instant: policyengine-uk
    writes the April rates over the whole labelled year. This corrects the
    tree mismatch identified by uk-data#475 and uses the spine's shared week
    conversion identified by uk-data#476, 52 weeks since María's ruling of
    2026-10-23 (uk-data#531).
    """

    dwp = _uk_dwp_tree(year)
    import policyengine_uk

    instant = f"{year}-01-01"
    return UKDWPDisabilityCategoryRates(
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
        source="policyengine-uk parameters (fiscal-converted gov.dwp tree) "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def uk_dwp_disability_flag_rates(year: int) -> UKDWPDisabilityFlagRates:
    """Read flag rates from fiscal-converted ``gov.dwp`` at ``year``.

    The parameter instant is the fiscal-year label instant: policyengine-uk
    writes the April rates over the whole labelled year. This keeps the tree
    correction from uk-data#475 and the spine's shared week conversion from
    uk-data#476 (52 weeks since 2026-10-23) aligned with the category
    derivation.
    """

    dwp = _uk_dwp_tree(year)
    import policyengine_uk

    instant = f"{year}-01-01"
    return UKDWPDisabilityFlagRates(
        aa_higher=_parameter_value(dwp.attendance_allowance.higher, instant),
        dla_sc_higher=_parameter_value(dwp.dla.self_care.higher, instant),
        pip_dl_enhanced=_parameter_value(dwp.pip.daily_living.enhanced, instant),
        instant=instant,
        source="policyengine-uk parameters (fiscal-converted gov.dwp tree) "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


@lru_cache
def _uk_dwp_tree(year: int):
    """Return the fiscal-converted DWP tree from one engine per year."""

    try:
        import policyengine_uk
    except ImportError as exc:
        raise ImportError(
            "UK DWP disability parameters require `uv sync --all-packages --extra uk`."
        ) from exc

    return policyengine_uk.CountryTaxBenefitSystem().parameters(year).gov.dwp


def _assert_frs_disability_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind both disability derivations to the reviewed policy-year contract."""

    _assert_closed_world_operations(
        stage,
        (
            (
                "derive",
                {
                    "parameters": "disability category thresholds from the fiscal-converted gov.dwp tree at the survey year",
                    "year_rule": YEAR_RULE,
                },
            ),
            (
                "derive",
                {
                    "parameters": "disability flags from the fiscal-converted gov.dwp tree at the survey year",
                    "year_rule": YEAR_RULE,
                },
            ),
            (
                "derive",
                {"parameters": CARER_LINK_PARAMETERS, "year_rule": YEAR_RULE},
            ),
        ),
    )


def add_frs_disability(
    frame: Frame,
    *,
    category_rates: UKDWPDisabilityCategoryRates,
    flag_rates: UKDWPDisabilityFlagRates,
    carer_min_hours: float | None = None,
    evidence: dict[str, object] | None = None,
) -> Frame:
    person = frame.table("person").copy()
    derived = derive_frs_disability(
        person, category_rates=category_rates, flag_rates=flag_rates
    )
    for column in FRS_DISABILITY_OUTPUT_COLUMNS:
        person[column] = derived[column].to_numpy()
    if carer_min_hours is not None:
        household = frame.table("household")
        weight_by_household = dict(
            zip(
                household["household_id"].to_numpy(),
                np.asarray(frame.weights_for("household").values, dtype=float),
                strict=True,
            )
        )
        link, link_evidence = derive_carer_link(
            person,
            weights=person["person_household_id"]
            .map(weight_by_household)
            .to_numpy(float),
            min_hours=carer_min_hours,
        )
        person[CARER_LINK_COLUMN] = link
        if evidence is not None:
            evidence["carer_link"] = link_evidence
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
    category_rates: UKDWPDisabilityCategoryRates,
    flag_rates: UKDWPDisabilityFlagRates,
) -> pd.DataFrame:
    values = pd.DataFrame(index=person.index)
    values["aa_category"] = _category(
        _amount(person, "attendance_allowance_reported"),
        (("LOWER", category_rates.aa_lower), ("HIGHER", category_rates.aa_higher)),
    )
    values["dla_sc_category"] = _category(
        _amount(person, "dla_sc_reported"),
        (
            ("LOWER", category_rates.dla_sc_lower),
            ("MIDDLE", category_rates.dla_sc_middle),
            ("HIGHER", category_rates.dla_sc_higher),
        ),
    )
    values["dla_m_category"] = _category(
        _amount(person, "dla_m_reported"),
        (
            ("LOWER", category_rates.dla_m_lower),
            ("HIGHER", category_rates.dla_m_higher),
        ),
    )
    values["pip_m_category"] = _category(
        _amount(person, "pip_m_reported"),
        (
            ("STANDARD", category_rates.pip_m_standard),
            ("ENHANCED", category_rates.pip_m_enhanced),
        ),
    )
    values["pip_dl_category"] = _category(
        _amount(person, "pip_dl_reported"),
        (
            ("STANDARD", category_rates.pip_dl_standard),
            ("ENHANCED", category_rates.pip_dl_enhanced),
        ),
    )
    total = sum(
        _amount(person, column) for column in UK_DISABILITY_FLAG_REPORTED_COLUMNS
    )
    dla_sc = _amount(person, "dla_sc_reported")
    aa = _amount(person, "attendance_allowance_reported")
    pip_dl = _amount(person, "pip_dl_reported")
    gap = WEEKS_IN_YEAR
    aa_higher = flag_rates.aa_higher * WEEKS_IN_YEAR - gap
    dla_sc_higher = flag_rates.dla_sc_higher * WEEKS_IN_YEAR - gap
    pip_dl_enhanced = flag_rates.pip_dl_enhanced * WEEKS_IN_YEAR - gap
    values["is_disabled_for_benefits"] = total > 0
    values["is_enhanced_disabled_for_benefits"] = (
        (aa >= aa_higher) | (dla_sc > dla_sc_higher) | (pip_dl >= pip_dl_enhanced)
    )
    # The tax credit severe disability condition (CTC Regs 2002 reg 8, WTC
    # Regs 2002 reg 17) on the categories derived above, so the stored flag
    # agrees with them by construction and with policyengine-uk's formula
    # (pe-uk#1946, uk-data#494): the DLA care component at the highest rate,
    # the PIP daily living component at the enhanced rate, or higher-rate
    # Attendance Allowance. Lower-rate Attendance Allowance does not count,
    # and FRS code 8 (the Armed Forces Compensation Scheme) cannot be told
    # apart from Armed Forces Independence Payment, so it no longer counts.
    values["is_severely_disabled_for_benefits"] = (
        (values["aa_category"] == "HIGHER")
        | (values["dla_sc_category"] == "HIGHER")
        | (values["pip_dl_category"] == "ENHANCED")
    ).to_numpy(dtype=bool)
    return values


def uk_dwp_carers_allowance_min_hours(year: int) -> float:
    """``gov.dwp.carers_allowance.min_hours`` at ``year``, as the carers stage reads it."""

    return _parameter_value(
        _uk_dwp_tree(year).carers_allowance.min_hours, f"{year}-01-01"
    )


class _CarerRoutes(NamedTuple):
    """Where each person stands on the carer link's routes."""

    ids: np.ndarray
    households: np.ndarray
    carer: np.ndarray
    outside: np.ndarray
    member_qualifies: np.ndarray
    hours: np.ndarray
    reported: np.ndarray


def _carer_routes(person: pd.DataFrame) -> _CarerRoutes:
    ids = person["person_id"].to_numpy(dtype=np.int64)
    households = person["person_household_id"].to_numpy(dtype=np.int64)
    mask = (
        person[FRS_CARE_RECIPIENTS_COLUMN].to_numpy(dtype=np.int64)
        if FRS_CARE_RECIPIENTS_COLUMN in person
        else np.zeros(len(person), dtype=np.int64)
    )
    qualifies = np.zeros(len(person), dtype=bool)
    for column, categories in CARER_LINK_QUALIFYING.items():
        qualifies |= person[column].astype(str).isin(categories).to_numpy()
    # A household member is keyed by household and FRS person number: the
    # spine's person_id is household id times 1,000 plus the number, and the
    # SPI and CGT copies shift both ids by a power of ten above every id, so
    # the number is person_id modulo 1,000 on every row.
    numbers = ids % 1000
    qualifying = set(
        zip(households[qualifies].tolist(), numbers[qualifies].tolist(), strict=True)
    )
    member_qualifies = np.zeros(len(person), dtype=bool)
    for number in range(1, FRS_CARE_OUTSIDE_BIT):
        holds = ((mask >> number) & 1) == 1
        if holds.any():
            member_qualifies[holds] |= np.asarray(
                [(int(h), number) in qualifying for h in households[holds]], dtype=bool
            )
    return _CarerRoutes(
        ids=ids,
        households=households,
        carer=mask != 0,
        outside=((mask >> FRS_CARE_OUTSIDE_BIT) & 1) == 1,
        member_qualifies=member_qualifies,
        hours=_amount(person, "care_hours").to_numpy(dtype=float),
        reported=_amount(person, "carers_allowance_reported").to_numpy(dtype=float) > 0,
    )


def carer_link_share(
    person: pd.DataFrame, *, weights: np.ndarray, min_hours: float
) -> tuple[float | None, int]:
    """The share the carer link draws outside carers at, and its households.

    Of the carers who care only within their household, whose care hours reach
    ``min_hours`` and who report no Carer's Allowance, the share by ``weights``
    whose main recipients include a qualifying member; None where none of them
    has weight. The second value is the distinct households that base rests
    on, for the receipt to band.
    """

    routes = _carer_routes(person)
    weights = np.asarray(weights, dtype=float)
    base = (
        routes.carer
        & ~routes.outside
        & (routes.hours >= float(min_hours))
        & ~routes.reported
    )
    base_weight = float(weights[base].sum())
    share = (
        float(weights[base & routes.member_qualifies].sum()) / base_weight
        if base_weight > 0.0
        else None
    )
    return share, len(set(routes.households[base].tolist()))


def derive_carer_link(
    person: pd.DataFrame,
    *,
    weights: np.ndarray,
    min_hours: float,
    share: tuple[float | None, int] | None = None,
    support_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """The carer link for every person, and its evidence (see :data:`CARER_LINK_COLUMN`).

    ``person`` carries ``person_id`` (the FRS person number is ``person_id``
    modulo 1,000 on spine rows and copies alike) and ``person_household_id``,
    the care carriers and the categories this stage derives; ``weights`` are
    the persons' household weights. ``share`` is the drawn route's share and
    the households it rests on, by default :func:`carer_link_share` of
    ``person``; the SPI channel passes the survey's own (``spi_income``).
    ``support_keys`` are what the evidence counts support in, by default each
    person's household; the SPI channel passes source households, so a survey
    household's copies count once. The evidence gives each route's records in
    rounded counts and support bands, withheld where one or two households
    carry them, and the share the drawn route takes.
    """

    routes = _carer_routes(person)
    weights = np.asarray(weights, dtype=float)
    if share is None:
        share = carer_link_share(person, weights=weights, min_hours=min_hours)
    share_value, share_households = share
    keys = routes.households if support_keys is None else np.asarray(support_keys)
    linked = routes.carer & routes.member_qualifies
    drawn_route = routes.carer & routes.outside & ~routes.member_qualifies
    household_route = routes.carer & ~routes.outside
    no_link = ~routes.carer & (routes.hours > 0)
    at_line = routes.hours >= float(min_hours)
    draws = stable_identity_uniforms(
        routes.ids, seed=CARER_LINK_SEED, salt=CARER_LINK_SALT
    )
    positions = np.asarray(draws, dtype=float).copy()
    if drawn_route.any():
        positions[drawn_route] = lineage_balanced_positions(
            at_line[drawn_route],
            positions[drawn_route],
            weights[drawn_route],
            identities=routes.ids[drawn_route],
            salt=CARER_LINK_SALT,
        )
    drawn = drawn_route & (positions < (share_value or 0.0))
    link = linked | drawn | no_link

    def counted(selected: np.ndarray) -> int | str:
        # Record counts are rounded, never exact, and withheld where one or
        # two households carry them (UKDS CD137 cl. 8).
        households = len(set(keys[selected].tolist()))
        if 0 < households < UK_DISCLOSURE_FLOOR:
            return UK_WITHHELD
        return rounded_count(int(selected.sum()))

    def route(selected: np.ndarray) -> dict[str, object]:
        return {
            "records": counted(selected),
            "support": support_band(len(set(keys[selected].tolist()))),
            "true": counted(selected & link),
        }

    evidence: dict[str, object] = {
        "column": CARER_LINK_COLUMN,
        "rule": CARER_LINK_PARAMETERS,
        "min_hours": float(min_hours),
        "share": (
            None
            if share_value is None
            else small_cell_share(share_value, share_households)
        ),
        "share_support": support_band(share_households),
        "routes_at_the_line": {
            "household_qualifying": route(
                at_line & household_route & routes.member_qualifies
            ),
            "household_not_qualifying": route(
                at_line & household_route & ~routes.member_qualifies
            ),
            "outside_with_a_qualifying_member": route(
                at_line & routes.carer & routes.outside & routes.member_qualifies
            ),
            "outside_drawn": route(at_line & drawn_route),
            "no_link": route(at_line & no_link),
        },
    }
    return link, evidence


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
