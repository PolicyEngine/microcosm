"""UK FRS household-grain stochastic assignments."""

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

FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS = (
    "household_owns_tv",
    "would_evade_tv_licence_fee",
    "main_residential_property_purchased_is_first_home",
    "property_purchased",
)
UK_HOUSEHOLD_DRAW_DECLARED_SEEDS = {
    output: 0 for output in FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS
}


@dataclass(frozen=True)
class UKFRSHouseholdDrawsStageTransform:
    """Whole-stage callable for UK household stochastic draws."""

    contract: UKTakeUpContract
    stage: SourceStageSpec

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_household_draws(frame, contract=self.contract)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS


def add_frs_household_draws(frame: Frame, *, contract: UKTakeUpContract) -> Frame:
    household = frame.table("household").copy()
    derived = derive_frs_household_draws(household, contract=contract)
    for column in FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS:
        household[column] = derived[column].to_numpy()
    result = uk_national_frame(
        person=frame.table("person").copy(),
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def derive_frs_household_draws(
    household: pd.DataFrame, *, contract: UKTakeUpContract
) -> pd.DataFrame:
    ids = household["household_id"].to_numpy()
    values = pd.DataFrame(index=household.index)
    for output, key in (
        ("household_owns_tv", "tv_ownership_rate"),
        ("would_evade_tv_licence_fee", "tv_licence_evasion_rate"),
        ("main_residential_property_purchased_is_first_home", "first_time_buyer_rate"),
        ("property_purchased", "property_purchase_rate"),
    ):
        values[output] = assign_binary_from_rate(
            _draws(ids, output), contract.rate(key)
        )
    return values


def _draws(ids: np.ndarray, output: str) -> np.ndarray:
    return stable_identity_uniforms(
        ids, seed=UK_HOUSEHOLD_DRAW_DECLARED_SEEDS[output], salt=output
    )
