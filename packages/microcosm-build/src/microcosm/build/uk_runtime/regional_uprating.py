"""UK regional property uprating stage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.property_localisation import RANK_BASIS_COLUMN
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.frame import Frame

#: The property total and the components of it the frame carries. The factor
#: is read off the main residence and applied to the total, so it is applied to
#: every component too: the total stays the sum of its parts (microcosm#1063).
UK_REGIONAL_PROPERTY_COMPONENT_REWRITES = (
    "owned_land",
    "other_residential_property_value",
    "non_residential_property_value",
)
UK_REGIONAL_PROPERTY_REWRITES = (
    "main_residence_value",
    "property_wealth",
    *UK_REGIONAL_PROPERTY_COMPONENT_REWRITES,
)


#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON = "Regional property uprating on the source spine: household weights pass through unchanged and total household mass is conserved."
#: The packaged regional reference the factor reads: each region's public
#: mean house price.
UK_REGIONAL_LAND_VALUES_RESOURCE = "regional_land_values.json"
#: The stage's operation, whose optional ``denominator`` names the column the
#: factor's mean is taken over (microcosm#931).
UK_REGIONAL_PROPERTY_UPRATING_OPERATION_KIND = "uprate_to_regional_reference"
#: The column the factor's mean is taken over when the operation declares no
#: denominator: the main residence the stage rewrites, as before
#: microcosm#931.
UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR = "main_residence_value"
#: The denominators the operation may declare: the main residence, or the WAS
#: chain's draw the spine's area home values keep beside the value they read
#: off the constituency's knots (``area_home_values``), so that the factor is
#: the one those values were divided by (microcosm#931).
UK_REGIONAL_PROPERTY_DENOMINATORS = (
    UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR,
    RANK_BASIS_COLUMN,
)


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
        denominator = declared_regional_reference_denominator(self.stage)
        if (
            RANK_BASIS_COLUMN in frame.table("household").columns
            and denominator == UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR
        ):
            # Form F's two halves come together (C-15): homes the WAS chain
            # read off their constituency's knots, divided by the factor on
            # the draw it keeps, land on the knots only when this stage
            # divides by that draw too (microcosm#931).
            raise ValueError(
                f"The regional property uprating reads households carrying "
                f"{RANK_BASIS_COLUMN!r}, the draw the WAS chain's area home "
                "values keep, but divides by main_residence_value: a stage "
                "after them declares that draw its denominator (microcosm#931)."
            )
        household = uprate_household_property_by_region(
            frame.table("household"),
            resource,
            denominator_column=denominator,
        )
        result = uk_national_frame(
            person=frame.table("person").copy(),
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()


def load_regional_land_values_resource() -> Mapping[str, Any]:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath(UK_REGIONAL_LAND_VALUES_RESOURCE)
        .read_text()
    )


def declared_regional_reference_denominator(stage: SourceStageSpec) -> str:
    """The column the stage's declared factor divides by.

    The ``uprate_to_regional_reference`` operation's ``denominator``, or the
    main residence where it declares none. Refused, by name: a denominator
    outside :data:`UK_REGIONAL_PROPERTY_DENOMINATORS`, and a stage declaring
    the operation more than once.
    """

    operations = [
        operation
        for operation in stage.operations
        if operation.kind == UK_REGIONAL_PROPERTY_UPRATING_OPERATION_KIND
    ]
    if len(operations) > 1:
        raise ValueError(
            f"The regional property uprating declares "
            f"{UK_REGIONAL_PROPERTY_UPRATING_OPERATION_KIND!r} more than once."
        )
    if not operations:
        return UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR
    denominator = operations[0].parameters.get(
        "denominator", UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR
    )
    if denominator not in UK_REGIONAL_PROPERTY_DENOMINATORS:
        raise ValueError(
            "The regional property uprating divides by one of "
            f"{list(UK_REGIONAL_PROPERTY_DENOMINATORS)}; the stage declares "
            "another denominator."
        )
    return str(denominator)


def _regional_house_prices(resource: Mapping[str, Any]) -> dict[str, float]:
    values = resource.get("values")
    if not isinstance(values, list):
        raise ValueError("regional land values resource must contain a values list.")
    return {
        str(entry["region"]): float(entry["avg_house_price"])
        for entry in values
        if isinstance(entry, Mapping)
    }


def _base_rows(household: pd.DataFrame) -> pd.Series:
    channel = support_channel_column("household")
    return (
        household[channel] == BASE_FRS_SUPPORT_CHANNEL
        if channel in household.columns
        else pd.Series(True, index=household.index)
    )


def _region_owners(household: pd.DataFrame, region: str) -> pd.Series:
    """The region's owner rows: a positive main residence, as declared."""

    region_mask = household["region"].astype(str) == region
    return region_mask & (
        pd.to_numeric(household["main_residence_value"], errors="coerce") > 0
    )


def regional_reference_factors(
    household: pd.DataFrame,
    resource: Mapping[str, Any],
    *,
    denominator_column: str = UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR,
) -> dict[str, float]:
    """Each region's factor: its public mean house price over a mean of its
    FRS-base owners.

    The owners are the region's rows with a positive ``main_residence_value``;
    the mean is the unweighted mean of their ``denominator_column``, summed in
    sorted order so that it does not depend on the table's row order. SPI
    support rows do not enter it. A region the resource lists with no FRS-base
    owner, or whose mean is not positive, has no factor. Returns the factors in
    the resource's order of regions.
    """

    required = {"region", "main_residence_value", denominator_column}
    missing = sorted(required - set(household.columns))
    if missing:
        raise KeyError(
            f"household table is missing property-uprating columns: {missing}"
        )
    hpi_prices = _regional_house_prices(resource)
    base_rows = _base_rows(household)
    factors: dict[str, float] = {}
    for region, hpi_price in hpi_prices.items():
        base_owners = _region_owners(household, region) & base_rows
        if not base_owners.any():
            continue
        # Sort before summing so the unweighted mean is independent of row
        # order: the stage is receipted bitwise under row permutation, and
        # float addition is not associative.
        owner_values = np.sort(
            household.loc[base_owners, denominator_column].to_numpy(dtype=float)
        )
        imputed_mean = float(owner_values.sum() / len(owner_values))
        if imputed_mean <= 0:
            continue
        factors[region] = hpi_price / imputed_mean
    return factors


def uprate_household_property_by_region(
    household: pd.DataFrame,
    resource: Mapping[str, Any],
    *,
    denominator_column: str = UK_REGIONAL_PROPERTY_DEFAULT_DENOMINATOR,
) -> pd.DataFrame:
    """Scale owner property values to region-level public house-price means.

    The factor for a region is its public mean house price over the unweighted
    mean ``denominator_column`` of FRS-base owners
    (:func:`regional_reference_factors`), the ``main_residence_value`` unless
    the stage declares otherwise. When the frame carries SPI support rows,
    which are imputed from higher incomes, they are uprated by the same factor
    but do not enter the mean. The factor scales the property total and every
    component of it the table carries, so an owner's ``property_wealth`` stays
    the sum of its components after the uprating.
    """

    required = {"region", "main_residence_value", "property_wealth", denominator_column}
    missing = sorted(required - set(household.columns))
    if missing:
        raise KeyError(
            f"household table is missing property-uprating columns: {missing}"
        )
    factors = regional_reference_factors(
        household, resource, denominator_column=denominator_column
    )
    result = household.copy()
    for region, factor in factors.items():
        owners_mask = _region_owners(result, region)
        for column in UK_REGIONAL_PROPERTY_REWRITES:
            if column in result.columns:
                result.loc[owners_mask, column] *= factor
    return result
