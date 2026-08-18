"""UK regional property uprating stage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame

UK_REGIONAL_PROPERTY_REWRITES = ("main_residence_value", "property_wealth")


@dataclass(frozen=True)
class UKRegionalPropertyUpratingStageTransform:
    """Whole-stage callable for regional property-value uprating."""

    stage: SourceStageSpec
    resource: Mapping[str, Any] | None = None

    def __call__(self, frame: Frame) -> Frame:
        resource = (
            self.resource
            if self.resource is not None
            else load_regional_land_values_resource()
        )
        household = uprate_household_property_by_region(
            frame.table("household"),
            resource,
        )
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

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()


def load_regional_land_values_resource() -> Mapping[str, Any]:
    return json.loads(
        files("microcosm.build.uk").joinpath("regional_land_values.json").read_text()
    )


def uprate_household_property_by_region(
    household: pd.DataFrame,
    resource: Mapping[str, Any],
) -> pd.DataFrame:
    """Scale owner property values to region-level public house-price means."""

    required = {"region", "main_residence_value", "property_wealth"}
    missing = sorted(required - set(household.columns))
    if missing:
        raise KeyError(
            f"household table is missing property-uprating columns: {missing}"
        )
    values = resource.get("values")
    if not isinstance(values, list):
        raise ValueError("regional land values resource must contain a values list.")
    hpi_prices = {
        str(entry["region"]): float(entry["avg_house_price"])
        for entry in values
        if isinstance(entry, Mapping)
    }
    result = household.copy()
    for region, hpi_price in hpi_prices.items():
        region_mask = result["region"].astype(str) == region
        owners_mask = region_mask & (
            pd.to_numeric(result["main_residence_value"], errors="coerce") > 0
        )
        if not owners_mask.any():
            continue
        imputed_mean = float(result.loc[owners_mask, "main_residence_value"].mean())
        if imputed_mean <= 0:
            continue
        factor = hpi_price / imputed_mean
        result.loc[owners_mask, "main_residence_value"] *= factor
        result.loc[owners_mask, "property_wealth"] *= factor
    return result
