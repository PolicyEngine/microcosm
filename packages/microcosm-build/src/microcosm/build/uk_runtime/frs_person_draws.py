"""UK FRS person-grain stochastic assignments.

``is_apprentice`` (microcosm#1372) has no FRS source: the 2022-23
to 2024-25 adult tables carry no apprenticeship variable, and ``TRAIN2``,
``NI3TRAIN`` and ``TRAINEE`` code government employment schemes, not
apprenticeships. policyengine-uk zero-rates employer Class 1 NICs for a
relevant apprentice under 25 (SSCBA 1992 s.9B), and a person under 21 already
gets the s.9A zero rate by age, so the flag changes employer NICs only at ages
21 to 24. The stage draws it there, for employees only, at DfE's published
participation rate for apprentices aged 19 to 24 in England per head of
population (``apprenticeship_participation_19_to_24``), converted to a rate
among the eligible employees at the frame's own household weights:

    p = min(1, r * W(age 21 to 24) / W(employees aged 21 to 24))

so the weighted flags among 21- to 24-year-olds equal r times their weighted
population. Ages under 21 and from 25 keep the engine default (False), which
leaves ``is_in_non_advanced_education`` (child benefit and Universal Credit
qualifying young person status), ``minimum_wage``, Care to Learn and the 16 to
19 Bursary Fund unchanged for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.frs_employment import FRS_EMPSTATI_EMPLOYMENT_STATUS
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
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
APPRENTICE_RATE_KEY = "apprenticeship_participation_19_to_24"
# SSCBA 1992 s.9A gives every earner under 21 the zero rate by age, and s.9B
# ends the apprentice zero rate at 25, so only ages 21 to 24 are drawn.
APPRENTICE_MIN_AGE = 21
APPRENTICE_MAX_AGE_EXCLUSIVE = 25
APPRENTICE_EMPLOYEE_STATUSES = ("FT_EMPLOYED", "PT_EMPLOYED")
if not set(APPRENTICE_EMPLOYEE_STATUSES) <= set(
    FRS_EMPSTATI_EMPLOYMENT_STATUS.values()
):
    raise RuntimeError("apprentice employee statuses must be FRS EMPSTATI statuses.")
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

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_person_draws(frame, contract=self.contract)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_PERSON_DRAW_OUTPUT_COLUMNS


def add_frs_person_draws(frame: Frame, *, contract: UKTakeUpContract) -> Frame:
    period = uk_time_period(frame)
    if int(str(period).split("-", maxsplit=1)[0]) != contract.build_year:
        raise ValueError(
            "frs_person_draws assign_period_constant requires the frame time "
            f"period {period!r} to equal contract build_year {contract.build_year}."
        )
    _require_tax_free_childcare_routed_share_variable()
    person = frame.table("person").copy()
    if APPRENTICE_OUTPUT in person.columns:
        raise ValueError(
            f"{APPRENTICE_OUTPUT!r} is already stored: frs_person_draws is its "
            "only writer."
        )
    derived = derive_frs_person_draws(
        person, contract=contract, person_weights=uk_person_household_weights(frame)
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
    return result


def derive_frs_person_draws(
    person: pd.DataFrame,
    *,
    contract: UKTakeUpContract,
    person_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return the stage's person draws.

    ``person_weights`` are each person's household weight; the stage passes
    the frame's. Without them every person weighs one, which keeps the draws
    themselves identity-keyed for callers that only check their stability.
    """

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
    values[APPRENTICE_OUTPUT] = derive_is_apprentice(
        person,
        rate=contract.rate(APPRENTICE_RATE_KEY, contract.build_year),
        person_weights=person_weights,
    )
    return values


def derive_is_apprentice(
    person: pd.DataFrame,
    *,
    rate: float,
    person_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Draw ``is_apprentice`` for employees aged 21 to 24.

    ``rate`` is apprentices per head of population at those ages. It becomes a
    rate among the eligible employees, ``rate * W(21 to 24) / W(eligible)``,
    capped at one, so the weighted flags equal ``rate`` times the weighted
    population aged 21 to 24 unless too few employees carry them. A person is
    flagged when eligible and their identity-keyed uniform is below that rate.
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
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    in_ages = (age >= APPRENTICE_MIN_AGE) & (age < APPRENTICE_MAX_AGE_EXCLUSIVE)
    employee = (
        person["employment_status"]
        .astype(str)
        .isin(APPRENTICE_EMPLOYEE_STATUSES)
        .to_numpy()
    )
    eligible = in_ages & employee
    eligible_mass = float(weights[eligible].sum())
    if eligible_mass <= 0:
        return np.zeros(len(person), dtype=bool)
    conditional = min(1.0, float(rate) * float(weights[in_ages].sum()) / eligible_mass)
    return eligible & assign_binary_from_rate(
        _draws(person["person_id"].to_numpy(), APPRENTICE_OUTPUT), conditional
    )


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


def _require_tax_free_childcare_routed_share_variable() -> None:
    """Fail before export when an installed UK engine cannot consume the input."""

    try:
        import policyengine_uk
    except ImportError:
        return
    variable = "tax_free_childcare_spend_routed_share"
    if variable not in policyengine_uk.CountryTaxBenefitSystem().variables:
        raise ValueError(
            f"Required variable {variable!r} is not defined in policyengine-uk."
        )


def _draws(ids: np.ndarray, output: str) -> np.ndarray:
    return stable_identity_uniforms(
        ids, seed=UK_PERSON_DRAW_DECLARED_SEEDS[output], salt=output
    )
