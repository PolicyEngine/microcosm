"""UK FRS person-grain stochastic assignments.

``is_apprentice`` (microcosm#1372) has no FRS source: the 2022-23
to 2024-25 adult tables carry no apprenticeship variable, and ``TRAIN2``,
``NI3TRAIN`` and ``TRAINEE`` code government employment schemes, not
apprenticeships. policyengine-uk zero-rates employer Class 1 NICs for a
relevant apprentice under 25 (SSCBA 1992 s.9B), and a person under 21 already
gets the s.9A zero rate by age, so the flag changes employer NICs only at ages
21 to 24. The stage draws it there, for employees only. Its target is r times
the weighted population aged 21 to 24 at the frame's own household weights,
where r (``apprentice_stock_19_to_24``) is DfE's England apprentices in
learning at the start of the academic year (participation less starts) per
head aged 19 to 24, a point-in-time count rather than the in-year one
(microcosm#1373). Eligible employees are flagged in the order of their
identity-keyed uniform until their weight is nearest the target, and the
stage's receipt records how near it came, and the employer NICs relief the
flags imply at the engine's own Class 1 parameters. Ages under 21 and from 25
keep the engine default (False), which leaves ``is_in_non_advanced_education``
(child benefit and Universal Credit qualifying young person status),
``minimum_wage``, Care to Learn and the 16 to 19 Bursary Fund unchanged for
them.

The flag is read twice in policyengine-uk: by the s.9B threshold (from 2.160.0,
policyengine-uk#2505) and by the apprentice minimum wage. Before 2.162.8 the
apprentice rate applied to every flagged person, so a flagged 21 to 24-year-old
lost the adult minimum wage, which moves the Tax-Free Childcare, extended
childcare and Childcare Offer for Wales income tests. From 2.162.8
(policyengine-uk#2683) the rate applies only under 19 or with
``is_in_first_year_of_apprenticeship``, which the build leaves at its default
(False), so the flagged keep their age-related minimum wage. The stage refuses
an installed engine without both readings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.frs_employment import FRS_EMPSTATI_EMPLOYMENT_STATUS
from microcosm.build.uk_runtime.frs_take_up import (
    geographic_clone_indices,
    geographic_clone_zero_rows,
    take_up_signal_outputs,
    uk_take_up_signal_receipt,
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
    distinct_count,
    record_source_households,
    support_band,
)
from microcosm.build.uk_runtime.take_up_contract import UKTakeUpContract
from microcosm.frame import Frame

FRS_PERSON_DRAW_OUTPUT_COLUMNS = (
    "would_claim_marriage_allowance",
    "would_claim_scp",
    "attends_private_school_random_draw",
    "tax_free_childcare_spend_routed_share",
    "is_apprentice",
)
FRS_PERSON_DRAW_NONNEGATIVE_OUTPUT_COLUMNS = (
    "attends_private_school_random_draw",
    "tax_free_childcare_spend_routed_share",
)
APPRENTICE_OUTPUT = "is_apprentice"
APPRENTICE_RATE_KEY = "apprentice_stock_19_to_24"
# SSCBA 1992 s.9A gives every earner under 21 the zero rate by age, and s.9B
# ends the apprentice zero rate at 25, so only ages 21 to 24 are drawn.
APPRENTICE_MIN_AGE = 21
APPRENTICE_MAX_AGE_EXCLUSIVE = 25
APPRENTICE_EMPLOYEE_STATUSES = ("FT_EMPLOYED", "PT_EMPLOYED")
if not set(APPRENTICE_EMPLOYEE_STATUSES) <= set(
    FRS_EMPSTATI_EMPLOYMENT_STATUS.values()
):
    raise RuntimeError("apprentice employee statuses must be FRS EMPSTATI statuses.")
# The engine readings is_apprentice needs: the s.9B threshold the flag exists
# for, and the minimum wage's split from it (policyengine-uk#2683).
APPRENTICE_ENGINE_VARIABLES = (
    "qualifies_for_apprentice_minimum_wage",
    "is_in_first_year_of_apprenticeship",
)
APPRENTICE_ENGINE_PARAMETER = (
    "gov.hmrc.national_insurance.class_1.thresholds."
    "apprentice_upper_secondary_threshold"
)
UK_PERSON_DRAW_DECLARED_SEEDS = {
    output: 0
    for output in FRS_PERSON_DRAW_OUTPUT_COLUMNS
    if output != "tax_free_childcare_spend_routed_share"
}


@dataclass(frozen=True)
class UKFRSPersonDrawsStageTransform:
    """Whole-stage callable for UK person stochastic draws."""

    contract: UKTakeUpContract
    stage: SourceStageSpec
    last_receipt: Mapping[str, object] | None = field(default=None, init=False)
    last_signal_receipt: Mapping[str, object] | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        result, receipt = add_frs_person_draws_with_receipt(
            frame, contract=self.contract
        )
        object.__setattr__(self, "last_receipt", receipt)
        # The Marriage Allowance and Scottish Child Payment flags at the stage's
        # weights, for the stage-time take-up signal gate (microcosm#1095).
        object.__setattr__(
            self,
            "last_signal_receipt",
            uk_take_up_signal_receipt(
                result,
                outputs=take_up_signal_outputs(FRS_PERSON_DRAW_OUTPUT_COLUMNS),
                contract=self.contract,
            ),
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_PERSON_DRAW_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None or self.last_signal_receipt is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "evidence": {
                "stage": "frs_person_draws",
                "apprentice_receipt": dict(self.last_receipt),
                "take_up_signal": dict(self.last_signal_receipt),
            }
        }


def add_frs_person_draws(frame: Frame, *, contract: UKTakeUpContract) -> Frame:
    return add_frs_person_draws_with_receipt(frame, contract=contract)[0]


def add_frs_person_draws_with_receipt(
    frame: Frame, *, contract: UKTakeUpContract
) -> tuple[Frame, Mapping[str, object]]:
    """The stage's frame and the receipt of its ``is_apprentice`` draw."""

    period = uk_time_period(frame)
    if int(str(period).split("-", maxsplit=1)[0]) != contract.build_year:
        raise ValueError(
            "frs_person_draws assign_period_constant requires the frame time "
            f"period {period!r} to equal contract build_year {contract.build_year}."
        )
    system = _installed_uk_system()
    _require_tax_free_childcare_routed_share_variable(system)
    schedule = apprentice_relief_schedule(system, period)
    person = frame.table("person").copy()
    if APPRENTICE_OUTPUT in person.columns:
        raise ValueError(
            f"{APPRENTICE_OUTPUT!r} is already stored: frs_person_draws is its "
            "only writer."
        )
    derived, receipt = derive_frs_person_draws_with_receipt(
        person,
        contract=contract,
        person_weights=uk_person_household_weights(frame),
        relief_schedule=schedule,
        person_sources=record_source_households(
            frame, "person", stage="frs_person_draws", before_copies=True
        ),
        person_clone_zero=geographic_clone_zero_rows(frame, "person"),
        # Each person's household's clone, which the apprentice walk meets its
        # target within; None on a frame without household_clone_index.
        person_clone_index=geographic_clone_indices(frame, "person"),
    )
    for column in FRS_PERSON_DRAW_OUTPUT_COLUMNS:
        person[column] = derived[column].to_numpy()
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        time_period=period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result, receipt


def derive_frs_person_draws(
    person: pd.DataFrame,
    *,
    contract: UKTakeUpContract,
    person_weights: np.ndarray,
    relief_schedule: Mapping[str, float] | None = None,
    person_sources: np.ndarray,
    person_clone_zero: np.ndarray | None = None,
    person_clone_index: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return the stage's person draws.

    ``person_weights`` are each person's household weight, which set
    ``is_apprentice``'s target, met within each geographic clone of
    ``person_clone_index`` (every person in one when not given);
    ``person_sources`` are each person's true source household, which its
    receipt counts support by, among geographic clone 0's people
    (``person_clone_zero``; where it is not given, clone 0 of
    ``person_clone_index``, else everyone).
    """

    return derive_frs_person_draws_with_receipt(
        person,
        contract=contract,
        person_weights=person_weights,
        relief_schedule=relief_schedule,
        person_sources=person_sources,
        person_clone_zero=person_clone_zero,
        person_clone_index=person_clone_index,
    )[0]


def derive_frs_person_draws_with_receipt(
    person: pd.DataFrame,
    *,
    contract: UKTakeUpContract,
    person_weights: np.ndarray,
    relief_schedule: Mapping[str, float] | None = None,
    person_sources: np.ndarray,
    person_clone_zero: np.ndarray | None = None,
    person_clone_index: np.ndarray | None = None,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """The stage's person draws and the receipt of its ``is_apprentice`` draw."""

    ids = person["person_id"].to_numpy()
    values = pd.DataFrame(index=person.index)
    values["would_claim_marriage_allowance"] = assign_binary_from_rate(
        _draws(ids, "would_claim_marriage_allowance"),
        contract.rate("marriage_allowance"),
    )
    scp_rates = np.where(
        pd.to_numeric(person["age"], errors="coerce").fillna(0).to_numpy() < 6,
        contract.rate("scp_under_6"),
        contract.rate("scp_6_plus"),
    )
    values["would_claim_scp"] = _draws(ids, "would_claim_scp") < scp_rates
    values["attends_private_school_random_draw"] = _draws(
        ids, "attends_private_school_random_draw"
    )
    entry = contract.entry("tax_free_childcare_spend_routed_share")
    if entry.raw.get("entity") != "person":
        raise ValueError(
            "tax_free_childcare_spend_routed_share contract entry must declare "
            "entity='person'."
        )
    values["tax_free_childcare_spend_routed_share"] = contract.rate(
        "tax_free_childcare_spend_routed_share", contract.build_year
    )
    apprentices = derive_is_apprentice(
        person,
        rate=contract.rate(APPRENTICE_RATE_KEY, contract.build_year),
        person_weights=person_weights,
        relief_schedule=relief_schedule,
        person_sources=person_sources,
        person_clone_zero=person_clone_zero,
        person_clone_index=person_clone_index,
    )
    values[APPRENTICE_OUTPUT] = apprentices.flags
    return values, {"build_year": contract.build_year, **apprentices.receipt}


@dataclass(frozen=True)
class UKApprenticeDraw:
    """``is_apprentice`` and the receipt of how the draw met its target."""

    flags: np.ndarray
    receipt: Mapping[str, object]


def derive_is_apprentice(
    person: pd.DataFrame,
    *,
    rate: float,
    person_weights: np.ndarray | None = None,
    relief_schedule: Mapping[str, float] | None = None,
    person_sources: np.ndarray,
    person_clone_zero: np.ndarray | None = None,
    person_clone_index: np.ndarray | None = None,
) -> UKApprenticeDraw:
    """Draw ``is_apprentice`` for employees aged 21 to 24.

    ``rate`` is apprentices per head of population at those ages, so the
    target is ``rate * W(21 to 24)``. Eligible employees are flagged in the
    order of their identity-keyed uniform until their weight is nearest the
    target (uc_take_up's rule, microcosm#1334), so the stage realizes it to
    within one person's weight rather than only in expectation. When the
    eligible weight falls short of the target every eligible employee is
    flagged and the receipt says the draw was cap-bound.

    The target is met within each geographic clone (microcosm#931).
    ``person_clone_index`` gives each person's clone (everyone is in one
    clone when it is not given), and each clone walks its own employees to
    the rate times its own weight aged 21 to 24. Clone 0 holds a one-clone
    spine's people at their weights over K, so it takes that spine's flags
    at any power-of-two K, which scales its cumulative weights exactly, and
    elsewhere except at a near-tie: where the target lies within rounding of
    the midpoint between two consecutive prefixes of the walk, or of the
    eligible weight, at which the walk flags everyone. One walk over every
    clone would interleave the clones' employees and stop once, so clone 0's
    flags would not be one clone's. The receipt's masses and realized weight
    are sums over the clones. Its target is the rate times the weight aged
    21 to 24 over every clone, and the draw is cap-bound where the eligible
    weight over every clone is no more than that target: the figures
    stage_health's apprentice_draw gate recomputes from the receipt, and at
    one clone the walk's own, bit for bit.

    Given a ``relief_schedule`` (the engine's Class 1 rate and its secondary
    and apprentice upper secondary thresholds, annual), the receipt also
    records the relief the flags imply: the rate on each flagged employee's
    pay between the two thresholds, weighted. Pay stands in for NIC earnings,
    so it is an estimate of the s.9B relief, not the engine's figure.

    The receipt counts support in ``person_sources``, each person's true
    source household (``receipt_counts.record_source_households``). A caller
    without the frame's lineage passes ``own_source_households`` itself, so
    no table is counted by its rows unawares (microcosm#931). The supports
    it prints are counted among geographic clone 0's people,
    ``person_clone_zero`` (where it is not given, the people
    ``person_clone_index`` puts in clone 0, else everyone): each clone flags
    its own employees, so the flagged sources over every clone would grow
    with the clone count. The masses are taken over every clone, so each is
    withheld where one or two source households carry it in clone 0's people
    or over every clone: clone 0 can flag none of the employees one or two
    households hold in the other clones.
    """

    if "employment_status" not in person.columns:
        raise ValueError(
            f"{APPRENTICE_OUTPUT} needs employment_status from frs_employment; "
            "it would otherwise flag people who are not employees."
        )
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError(f"{APPRENTICE_RATE_KEY} must lie in [0, 1]; got {rate}.")
    weights = (
        np.ones(len(person), dtype=float)
        if person_weights is None
        else np.asarray(person_weights, dtype=float)
    )
    if weights.shape != (len(person),):
        raise ValueError("person_weights must hold one weight per person.")
    clones = _clone_rows(person_clone_index, len(person))
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    in_ages = (age >= APPRENTICE_MIN_AGE) & (age < APPRENTICE_MAX_AGE_EXCLUSIVE)
    employee = (
        person["employment_status"]
        .astype(str)
        .isin(APPRENTICE_EMPLOYEE_STATUSES)
        .to_numpy()
    )
    eligible = in_ages & employee
    draws = _draws(person["person_id"].to_numpy(), APPRENTICE_OUTPUT)
    population_mass = float(weights[in_ages].sum())
    eligible_mass = float(weights[eligible].sum())
    flags = np.zeros(len(person), dtype=bool)
    for rows in clones:
        clone_eligible = eligible & rows
        threshold = _nearest_target_threshold(
            float(rate) * float(weights[in_ages & rows].sum()),
            draws=draws[clone_eligible],
            weights=weights[clone_eligible],
        )
        flags |= clone_eligible & (draws < threshold)
    # The receipt's target and cap bound are taken over every clone, as
    # stage_health's apprentice_draw gate recomputes them from the receipt's
    # masses; at one clone they are the walk's own, bit for bit.
    target = float(rate) * population_mass
    realized = float(weights[flags].sum())
    sources = np.asarray(person_sources)
    if sources.shape != (len(person),):
        raise ValueError("person_sources must hold one source per person.")
    clone_zero = _clone_zero_people(person_clone_zero, person_clone_index, len(person))
    population_support = distinct_count(sources, in_ages & clone_zero, weights)
    eligible_support = distinct_count(sources, eligible & clone_zero, weights)
    flagged_support = distinct_count(sources, flags & clone_zero, weights)
    # The masses are taken over every clone, so a mass is withheld where one
    # or two source households carry it in clone 0 or over every clone.
    population_withheld = _thin(population_support) or _thin(
        distinct_count(sources, in_ages, weights)
    )
    eligible_withheld = _thin(eligible_support) or _thin(
        distinct_count(sources, eligible, weights)
    )
    flagged_withheld = _thin(flagged_support) or _thin(
        distinct_count(sources, flags, weights)
    )
    receipt: dict[str, object] = {
        "output": APPRENTICE_OUTPUT,
        "rate_key": APPRENTICE_RATE_KEY,
        "rate": float(rate),
        "ages": [APPRENTICE_MIN_AGE, APPRENTICE_MAX_AGE_EXCLUSIVE - 1],
        "eligible_statuses": list(APPRENTICE_EMPLOYEE_STATUSES),
        "population_mass": population_mass,
        "eligible_mass": eligible_mass,
        "target": target,
        "cap_bound": bool(target > 0.0 and eligible_mass <= target),
        "population_support": support_band(population_support),
        "eligible_support": support_band(eligible_support),
        "flagged_support": support_band(flagged_support),
        "realized_mass": realized,
        "realization_deviation": realized - target,
    }
    # A weighted total carried by one or two households is withheld, with the
    # figures that would give it back: the target is the rate times the
    # population, and the deviation is the flagged weight less the target.
    if population_withheld:
        receipt.update(
            population_mass=UK_WITHHELD,
            target=UK_WITHHELD,
            realization_deviation=UK_WITHHELD,
        )
    if eligible_withheld:
        receipt["eligible_mass"] = UK_WITHHELD
    if relief_schedule is not None and "employment_income" in person.columns:
        pay = pd.to_numeric(person["employment_income"], errors="coerce")
        relieved = np.clip(
            np.minimum(
                pay.fillna(0.0).to_numpy(dtype=float),
                float(relief_schedule["apprentice_upper_secondary_threshold"]),
            )
            - float(relief_schedule["secondary_threshold"]),
            0.0,
            None,
        )
        receipt["relief_schedule"] = dict(relief_schedule)
        receipt["implied_relief"] = float(
            relief_schedule["rate"] * (weights[flags] * relieved[flags]).sum()
        )
    if flagged_withheld:
        receipt.update(realized_mass=UK_WITHHELD, realization_deviation=UK_WITHHELD)
        if "implied_relief" in receipt:
            receipt["implied_relief"] = UK_WITHHELD
    return UKApprenticeDraw(flags=flags, receipt=receipt)


def _thin(support: int) -> bool:
    """Whether one or two source households carry a figure: it is withheld."""

    return 0 < support < UK_DISCLOSURE_FLOOR


def _clone_rows(clone_index: np.ndarray | None, rows: int) -> tuple[np.ndarray, ...]:
    """Each geographic clone's rows of ``rows`` people, in sorted clone order.

    A ``clone_index`` that is None or holds one clone gives one mask of every
    row, so a one-clone frame's walk runs on its rows as a frame without the
    index does. The index must hold one integer per person; the refusals name
    no count, clone or row.
    """

    if clone_index is None:
        return (np.ones(rows, dtype=bool),)
    labels = np.asarray(clone_index)
    if labels.shape != (rows,):
        raise ValueError("person_clone_index must hold one clone per person.")
    if labels.dtype.kind not in "iu":
        raise TypeError("person_clone_index must hold an integer clone per person.")
    clones = np.unique(labels)
    if clones.size <= 1:
        return (np.ones(rows, dtype=bool),)
    return tuple(labels == clone for clone in clones)


def _clone_zero_people(
    clone_zero: np.ndarray | None, clone_index: np.ndarray | None, rows: int
) -> np.ndarray:
    """The people the receipt counts support among: ``clone_zero``, or where
    it is not given the people ``clone_index`` puts in clone 0, or everyone.
    Both given, they must agree; the refusals name no count or row."""

    if clone_zero is None:
        if clone_index is None:
            return np.ones(rows, dtype=bool)
        return np.asarray(clone_index) == 0
    flags = np.asarray(clone_zero, dtype=bool)
    if flags.shape != (rows,):
        raise ValueError("person_clone_zero must hold one flag per person.")
    if clone_index is not None and not np.array_equal(
        flags, np.asarray(clone_index) == 0
    ):
        raise ValueError(
            "person_clone_zero must flag exactly the people whose "
            "person_clone_index is zero."
        )
    return flags


def _nearest_target_threshold(
    need: float, *, draws: np.ndarray, weights: np.ndarray
) -> float:
    """The uniform below which eligible people are flagged.

    People are taken in the order of their uniform until their weight is
    nearest ``need``; flagging no one is a candidate too, when ``need`` sits
    nearer zero than the first person's weight. When the eligible weight is no
    more than ``need``, everyone eligible is flagged.
    """

    if need <= 0.0 or draws.size == 0:
        return 0.0
    if float(weights.sum()) <= need:
        return 1.0
    order = np.argsort(draws, kind="stable")
    cumulative = np.cumsum(weights[order])
    last = min(int(np.searchsorted(cumulative, need)), len(order) - 1)
    if last > 0 and abs(cumulative[last - 1] - need) < abs(cumulative[last] - need):
        last -= 1
    if last == 0 and abs(need) < abs(cumulative[0] - need):
        return float(draws[order[0]])
    return float(np.nextafter(draws[order[last]], 2.0))


def uk_person_household_weights(frame: Frame) -> np.ndarray:
    household_weights = frame.weights_for("household")
    weight_by_household = dict(
        zip(
            frame.table("household")["household_id"].to_numpy(),
            np.asarray(household_weights.values, dtype=float),
            strict=True,
        )
    )
    return (
        frame.table("person")["person_household_id"]
        .map(weight_by_household)
        .to_numpy(dtype=float)
    )


def _installed_uk_system():
    """The installed UK engine's tax-benefit system, or None without one."""

    try:
        import policyengine_uk
    except ImportError:
        return None
    return policyengine_uk.CountryTaxBenefitSystem()


def _require_tax_free_childcare_routed_share_variable(system=None) -> None:
    """Fail before export when an installed UK engine cannot consume the input."""

    if system is None:
        return
    variable = "tax_free_childcare_spend_routed_share"
    if variable not in system.variables:
        raise ValueError(
            f"Required variable {variable!r} is not defined in policyengine-uk."
        )


def apprentice_relief_schedule(system, period) -> dict[str, float | int] | None:
    """The engine's Class 1 schedule for the flags' relief, or None without one.

    Refuses an engine that would misread ``is_apprentice``: one without the
    s.9B threshold (policyengine-uk before 2.160.0) or without the minimum
    wage's split from the flag (before 2.162.8, policyengine-uk#2683), which
    would give every flagged employee the apprentice minimum wage. The values
    are the ones the engine charges in the stage's year, at its start, as a
    year's ``parameters(period)`` reads them; weekly thresholds are annualized
    over 52 weeks as ``secondary_class_1_threshold`` does.
    """

    if system is None:
        return None
    missing = [
        name for name in APPRENTICE_ENGINE_VARIABLES if name not in system.variables
    ]
    if missing:
        raise ValueError(
            f"{APPRENTICE_OUTPUT} needs policyengine-uk 2.162.8 or later, which "
            "gives the apprentice minimum wage only under 19 or in the first year "
            f"(policyengine-uk#2683); the installed engine lacks {missing}."
        )
    year = int(str(period).split("-", maxsplit=1)[0])
    parameters = system.parameters(f"{year}-01-01")
    class_1 = parameters.gov.hmrc.national_insurance.class_1
    if not hasattr(class_1.thresholds, "apprentice_upper_secondary_threshold"):
        raise ValueError(
            f"{APPRENTICE_OUTPUT} needs the installed engine's "
            f"{APPRENTICE_ENGINE_PARAMETER} (policyengine-uk 2.160.0 or later)."
        )
    secondary = class_1.employer_scale.components[1]
    return {
        "period": year,
        "rate": float(secondary.rate),
        "secondary_threshold": float(secondary.threshold) * 52.0,
        "apprentice_upper_secondary_threshold": float(
            class_1.thresholds.apprentice_upper_secondary_threshold
        )
        * 52.0,
    }


def _draws(ids: np.ndarray, output: str) -> np.ndarray:
    return stable_identity_uniforms(
        ids, seed=UK_PERSON_DRAW_DECLARED_SEEDS[output], salt=output
    )
