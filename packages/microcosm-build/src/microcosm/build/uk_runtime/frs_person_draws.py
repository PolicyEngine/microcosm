"""UK FRS person-grain stochastic assignments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    stable_identity_uniforms,
)
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
)
FRS_PERSON_DRAW_NONNEGATIVE_OUTPUT_COLUMNS = (
    "attends_private_school_random_draw",
    "tax_free_childcare_spend_routed_share",
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
    derived = derive_frs_person_draws(person, contract=contract)
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
    person: pd.DataFrame, *, contract: UKTakeUpContract
) -> pd.DataFrame:
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
    return values


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
