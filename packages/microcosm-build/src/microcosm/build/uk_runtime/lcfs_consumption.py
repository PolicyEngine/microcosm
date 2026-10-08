"""UK LCFS consumption imputation stage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.bus_fare_pricing import (
    PRICE_BUS_JOURNEYS_KIND,
    UK_DFT_BUS_JOURNEYS_RESOURCE,
    bus_fare_prices,
    price_bus_journeys,
)
from microcosm.build.uk_runtime.donor_uprating import (
    apply_donor_uprating,
    donor_uprating_factors,
    uprating_operation,
)
from microcosm.build.uk_runtime.energy_pricing import (
    DISCONNECT_IDENTITY_UNIFORM_ORDER,
    DISCONNECT_LOWEST_DRAWN_GAS_FIRST,
    ELECTRICITY_KWH,
    GAS_CONNECTED_POSITIVE_SPEND,
    GAS_CONNECTED_PUBLISHED_METER_SHARE,
    GAS_KWH,
    PRICE_DOMESTIC_ENERGY_KIND,
    UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
    UK_NEED_ENERGY_FACTS_RESOURCE,
    UK_QEP_ENERGY_PRICES_RESOURCE,
    EnergyPrices,
    NeedMargins,
    impose_gas_connection,
    kwh_to_spend,
    need_margins_from_facts,
    pricing_operation,
    published_energy_level,
    published_gas_connected_shares,
    qep_prices,
    rake_energy_kwh,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    UKSupportClipResult,
    support_clip_to_donor_with_receipt,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

WEEKS_IN_YEAR = 365.25 / 7
LCFS_HOUSEHOLD_FILENAME = "dvhh_ukanon_v2_2023.tab"
LCFS_HOUSEHOLD_SHA256 = (
    "6e78f0914be38e63853165486d641cbd790753cc471086210c6f672bfa18ca72"
)
LCFS_HOUSEHOLD_SIZE_BYTES = 22_812_887
LCFS_PERSON_FILENAME = "dvper_ukanon_202324_2023.tab"
LCFS_PERSON_SHA256 = "f32d54d83cdecf023f0ac73530be3a99372099b596e0106a56eae42a64929e50"
LCFS_PERSON_SIZE_BYTES = 6_545_146

UK_LCFS_CONSUMPTION_DECLARED_SEEDS = {"lcfs_consumption": 0}

LCFS_REGIONS: Mapping[int, str] = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}
LCFS_TENURE_MAP: Mapping[int, str] = {
    1: "RENT_FROM_COUNCIL",
    2: "RENT_FROM_HA",
    3: "RENT_PRIVATELY",
    4: "RENT_PRIVATELY",
    5: "OWNED_WITH_MORTGAGE",
    6: "OWNED_WITH_MORTGAGE",
    7: "OWNED_OUTRIGHT",
    8: "RENT_PRIVATELY",
}
LCFS_ACCOMM_MAP: Mapping[int, str] = {
    1: "HOUSE_DETACHED",
    2: "HOUSE_SEMI_DETACHED",
    3: "HOUSE_TERRACED",
    4: "FLAT",
    5: "FLAT",
    6: "MOBILE",
    7: "HOUSE_DETACHED",
    8: "OTHER",
}
HOUSEHOLD_LCFS_RENAMES = {
    "g018": "household_adult_count",
    "g019": "household_child_count",
    "gorx": "region",
    "a124": "num_vehicles",
    "p389p": "hbai_household_net_income",
    "p344p": "household_gross_income",
    "weighta": "household_weight",
}
PERSON_LCFS_RENAMES = {
    "b303p": "employment_income",
    "b3262p": "self_employment_income",
    "p049p": "private_pension_income",
}
CONSUMPTION_VARIABLE_RENAMES = {
    "p601": "food_and_non_alcoholic_beverages_consumption",
    "p602": "alcohol_and_tobacco_consumption",
    "p603": "clothing_and_footwear_consumption",
    "p604": "housing_water_and_electricity_consumption",
    "p605": "household_furnishings_consumption",
    "p606": "health_consumption",
    "p607": "transport_consumption",
    "p608": "communication_consumption",
    "p609": "recreation_consumption",
    "p610": "education_consumption",
    "p611": "restaurants_and_hotels_consumption",
    "p612": "miscellaneous_consumption",
    "c72211": "petrol_spending",
    "c72212": "diesel_spending",
    "p537": "domestic_energy_consumption",
}
#: The twelve COICOP division totals (LCFS p601-p612), in division order: the
#: columns policyengine-uk's ``consumption`` sums and its VAT base reads.
UK_LCFS_COICOP_DIVISION_COLUMNS = tuple(
    CONSUMPTION_VARIABLE_RENAMES[f"p6{division:02d}"] for division in range(1, 13)
)
BUS_FARE_LCFS_CODES = ("c73212", "c73213", "c73214")
#: Cars and vans available to the household: LCFS ``a124`` on the donor, the
#: was_wealth QRF draw of WAS ``vcarnr8`` on the recipient (the FRS carries no
#: vehicle variable). Both sides are rounded and clipped to this closed range so
#: the consumption QRF sees one predictor scale (microcosm#890 A).
UK_LCFS_VEHICLE_COUNT_RANGE = (0, 5)
#: Vendored Chronicle resources this stage reads; the declaration must name
#: these and no other (the vendor register lists this module as their consumer).
UK_LCFS_ROAD_FUEL_RESOURCE = "road_fuel_anchors.json"
UK_LCFS_LICENSED_CARS_RESOURCE = "licensed_cars_fuel_type.json"
UK_LCFS_DFT_BUS_VALUE_RESOURCE = "dft_bus_value_anchors.json"
# Re-exported for the allowed-resource set; the pricing module is the reader
# the register names.
UK_LCFS_DFT_BUS_JOURNEYS_RESOURCE = UK_DFT_BUS_JOURNEYS_RESOURCE
UK_LCFS_DEVOLVED_BUS_FINANCE_RESOURCE = "devolved_bus_finance.json"
#: ONS Consumer Trends household spending, the road-fuel level (COICOP 07.2.2).
UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE = "ons_household_expenditure_facts.json"
UK_LCFS_VENDORED_RESOURCES = (
    UK_LCFS_ROAD_FUEL_RESOURCE,
    UK_LCFS_LICENSED_CARS_RESOURCE,
    UK_LCFS_DFT_BUS_VALUE_RESOURCE,
    UK_LCFS_DFT_BUS_JOURNEYS_RESOURCE,
    UK_LCFS_DEVOLVED_BUS_FINANCE_RESOURCE,
    UK_NEED_ENERGY_FACTS_RESOURCE,
    UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
    UK_QEP_ENERGY_PRICES_RESOURCE,
    UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE,
)
#: The household road-fuel columns the ``level_road_fuel`` step scales.
UK_LCFS_ROAD_FUEL_COLUMNS = ("petrol_spending", "diesel_spending")
#: Columns a declared rake or pricing step levels (energy in kWh to the NEED
#: shape at the DESNZ level; bus fares as journeys times the published
#: yield, microcosm#930; road fuel to ONS household spending, microcosm#1113);
#: the committed support bounds leave them alone.
UK_LCFS_RAKED_COLUMNS = frozenset(
    {
        "electricity_consumption",
        "gas_consumption",
        "domestic_energy_consumption",
        "bus_fare_spending",
        *UK_LCFS_ROAD_FUEL_COLUMNS,
    }
)
LEVEL_ROAD_FUEL_KIND = "level_road_fuel"
#: The LCFS diary lines inside COICOP 07.2.2 beside petrol (c72211) and diesel
#: (c72212): other motor fuels and oils, which ONS's 07.2.2 also carries.
UK_LCFS_OTHER_ROAD_FUEL_CODES = ("c72213",)
UK_LCFS_ROAD_FUEL_OTHER_SHARE_RULE = "lcfs_donor_other_fuels_share"
REDRAW_ZERO_ROAD_FUEL_KIND = "redraw_zero_road_fuel"
UK_LCFS_ROAD_FUEL_REDRAW_RULE = "positive_total_then_petrol_share"
#: The redraw's chain targets: the household's road-fuel total, positive by
#: construction of the training set, then the petrol share of it (zero for a
#: diesel-only household).
UK_LCFS_ROAD_FUEL_TOTAL = "road_fuel_total"
UK_LCFS_PETROL_SHARE = "petrol_share_of_road_fuel"
UK_LCFS_ICE_SHARE_RULE = "one_minus_zero_emission_share"
#: How the litres audit reads its per-fuel ratios until a per-fuel cars benchmark
#: exists: only the all-fuel total compares like with like.
UK_LITRES_AUDIT_PER_FUEL_BASIS = "uniform cars share, not a per-fuel benchmark"
#: The litres audit (microcosm#890 C7) reads these vendored concepts beside the
#: declared litre-proxy price concepts: HMRC clearances are all road users, the
#: OBR receipts split names the cars share of them.
UK_HMRC_LITRES_CONCEPTS = {
    "petrol_spending": "hmrc.hydrocarbon_oils.total_petrol_quantity",
    "diesel_spending": "hmrc.hydrocarbon_oils.total_diesel_quantity",
}
UK_OBR_FUEL_DUTY_RECEIPTS_CONCEPT = "obr.fuel_duty.receipts"
UK_OBR_CARS_CATEGORY = "cars"
UK_OBR_TOTAL_CATEGORY = "total"
#: Columns whose level a declared rake sets; the support clip never touches them.
UK_LCFS_DEFAULT_SUPPORT_CLIP_EXEMPT = frozenset(
    {"electricity_consumption", "gas_consumption", "domestic_energy_consumption"}
)
#: The LCFS household's adults (G018) and children (G019), by the age-18 split
#: the survey counts them with. The recipient counts the same split from
#: ``age`` rather than the engine's person flags ``is_adult`` and ``is_child``,
#: which policyengine-uk#1896 deprecates (uk-data#486, microcosm#1095).
UK_LCFS_HOUSEHOLD_COUNT_ADULT_AGE = 18
UK_LCFS_HOUSEHOLD_COUNT_PREDICTORS = ("household_adult_count", "household_child_count")
UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS = (
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "hbai_household_net_income",
)
UK_LCFS_CONSUMPTION_PREDICTORS = (
    "household_adult_count",
    "household_child_count",
    "region",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "hbai_household_net_income",
    "tenure_type",
    "accommodation_type",
    "num_vehicles",
)
UK_LCFS_CONSUMPTION_TARGET_COLUMNS = (
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
    "petrol_spending",
    "diesel_spending",
    "bus_fare_spending",
    "domestic_energy_consumption",
    "electricity_consumption",
    "gas_consumption",
)
RECOMPOSE_FROM_REMAINDER_KIND = "recompose_from_remainder"
#: The COICOP totals that contain columns the stage levels (microcosm#1113):
#: each maps to the chain draws subtracted from its own draw to leave its
#: remainder, and the levelled columns added back. The chain draws each part
#: after its total, so the split is the chain's own. Housing (p604) nets the
#: drawn electricity and gas (the chain's domestic-energy target is the diary's
#: electricity plus gas), so the liquid and solid fuels no column carries stay
#: in its remainder; transport (p607) nets the drawn petrol and diesel (c72211,
#: c72212) and keeps other motor fuels.
UK_LCFS_RECOMPOSED_PARENTS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "housing_water_and_electricity_consumption": (
        ("domestic_energy_consumption",),
        ("electricity_consumption", "gas_consumption"),
    ),
    "transport_consumption": (
        ("petrol_spending", "diesel_spending"),
        ("petrol_spending", "diesel_spending"),
    ),
}
#: The chain draws a recomposition subtracts, kept before any step re-levels them.
UK_LCFS_RECOMPOSED_DRAWN_COLUMNS = tuple(
    dict.fromkeys(
        column
        for subtracted, _ in UK_LCFS_RECOMPOSED_PARENTS.values()
        for column in subtracted
    )
)
UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS = (
    *UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
    "has_fuel_consumption",
)
UK_LCFS_CONSUMPTION_NONNEGATIVE_OUTPUT_COLUMNS = UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS
UK_LCFS_CONSUMPTION_FIT_NAME = "uk_lcfs_2023_24_consumption"
UK_LCFS_CONSUMPTION_STAGE_NAME = "lcfs_consumption"


@dataclass
class UKLCFSConsumptionResult:
    """Transformed frame and donor-support clip receipt."""

    frame: Frame
    support_clip: UKSupportClipReceipt
    donor_uprating: Mapping[str, Any] | None = None
    fuel_flag: Mapping[str, Any] | None = None
    bus_pricing: Mapping[str, Any] | None = None
    energy_pricing: Mapping[str, Any] | None = None
    energy_rake: Mapping[str, Any] | None = None
    fuel_litres_audit: Mapping[str, Any] | None = None
    donor_floor: Mapping[str, Any] | None = None
    road_fuel_incidence: Mapping[str, Any] | None = None
    road_fuel_level: Mapping[str, Any] | None = None
    recomposed_totals: Mapping[str, Any] | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
            "stage": UK_LCFS_CONSUMPTION_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
        }
        if self.donor_uprating is not None:
            evidence["donor_uprating"] = dict(self.donor_uprating)
        if self.fuel_flag is not None:
            evidence["has_fuel_consumption"] = dict(self.fuel_flag)
        if self.bus_pricing is not None:
            evidence["bus_pricing"] = dict(self.bus_pricing)
        if self.energy_pricing is not None:
            evidence["energy_pricing"] = dict(self.energy_pricing)
        if self.energy_rake is not None:
            evidence["energy_rake"] = dict(self.energy_rake)
        if self.fuel_litres_audit is not None:
            evidence["fuel_litres_audit"] = dict(self.fuel_litres_audit)
        if self.donor_floor is not None:
            evidence["donor_floor"] = dict(self.donor_floor)
        if self.road_fuel_incidence is not None:
            evidence["road_fuel_incidence"] = dict(self.road_fuel_incidence)
        if self.road_fuel_level is not None:
            evidence["road_fuel_level"] = dict(self.road_fuel_level)
        if self.recomposed_totals is not None:
            evidence["recomposed_totals"] = dict(self.recomposed_totals)
        return evidence


#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON = "LCFS consumption imputation on the source spine: household weights pass through unchanged and total household mass is conserved."


@dataclass
class UKLCFSConsumptionStageTransform:
    """Whole-stage callable for LCFS-trained consumption imputation."""

    stage: SourceStageSpec
    engine: object
    lcfs_hh_tab_path: str | Path | None = None
    lcfs_person_tab_path: str | Path | None = None
    lcfs_household: pd.DataFrame | None = None
    lcfs_person: pd.DataFrame | None = None
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_result: UKLCFSConsumptionResult | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        lcfs_household = (
            self.lcfs_household
            if self.lcfs_household is not None
            else read_pinned_tab(
                _require_path(self.lcfs_hh_tab_path),
                _artifact(self.stage, "lcfs_household_tab"),
            )
        )
        lcfs_person = (
            self.lcfs_person
            if self.lcfs_person is not None
            else read_pinned_tab(
                _require_path(self.lcfs_person_tab_path),
                _artifact(self.stage, "lcfs_person_tab"),
            )
        )
        uprating_factors, uprating_receipt = lcfs_donor_uprating(self.stage)
        energy = lcfs_energy_pricing(self.stage)
        donor = clean_lcfs_consumption_table(
            lcfs_person,
            lcfs_household,
            uprating=uprating_factors,
            energy=energy,
            donor_rake_iterations=_donor_energy_rake_iterations(self.stage),
        )
        donor, donor_floor_receipt = floor_negative_donor_consumption(
            donor, floor=_donor_consumption_floor(self.stage)
        )
        ice_share, ice_share_receipt = lcfs_ice_share(self.stage)
        recipient = recipient_predictors(frame, self.engine)
        recipient["has_fuel_consumption"] = assign_recipient_has_fuel(
            frame,
            rate=ice_share,
            seed=_operation_seed(self.stage, "assign_binary_from_rate"),
        )
        weights = frame.weights_for("household").values
        fuel_flag_receipt = fuel_flag_evidence(
            donor,
            recipient,
            recipient_weights=weights,
            ice_share_receipt=ice_share_receipt,
        )
        imputation = impute_lcfs_consumption(
            donor,
            recipient,
            seed=_operation_seed(self.stage, "fit_weighted_qrf_chain"),
            n_estimators=_qrf_n_estimators(self.stage),
        )
        clip_result = support_clip_to_donor(
            imputation.draws, donor, exempt=support_clip_exempt(self.stage)
        )
        household_draws = clip_result.clipped
        # The chain's own draws of the parts later steps re-level, for the
        # recomposed totals (microcosm#1113).
        drawn_parts = household_draws[list(UK_LCFS_RECOMPOSED_DRAWN_COLUMNS)].copy()
        energy_rake_receipt = None
        if energy is not None:
            household_draws, energy_rake_receipt = rake_recipient_energy(
                household_draws,
                energy=energy,
                region=recipient["region"].astype(str).to_numpy(),
                income=recipient["household_gross_income"].to_numpy(dtype=float),
                tenure=recipient["tenure_type"].astype(str).to_numpy(),
                accommodation=recipient["accommodation_type"].astype(str).to_numpy(),
                weights=weights,
                iterations=_recipient_energy_rake_iterations(self.stage),
                identity=frame.table("household")["household_id"].to_numpy(),
            )
        household_draws, bus_pricing_receipt = lcfs_bus_fare_pricing(
            self.stage, household_draws, frame
        )
        household_draws["domestic_energy_consumption"] = (
            household_draws["electricity_consumption"]
            + household_draws["gas_consumption"]
        )
        household_draws.loc[
            ~recipient["has_fuel_consumption"].astype(bool),
            ["petrol_spending", "diesel_spending"],
        ] = 0.0
        incidence = lcfs_road_fuel_incidence(
            self.stage,
            household_draws,
            donor=donor,
            recipient=recipient,
            household_ids=frame.table("household")["household_id"].to_numpy(),
            weights=weights,
        )
        household_draws = incidence.draws
        household_draws, road_fuel_level_receipt = lcfs_road_fuel_level(
            self.stage, household_draws, weights=weights, lcfs_household=lcfs_household
        )
        household_draws, recomposed_totals_receipt = lcfs_recompose_from_remainder(
            self.stage, household_draws, drawn=drawn_parts, weights=weights
        )
        litres_audit = fuel_litres_audit(
            household_draws, weights=weights, stage=self.stage
        )
        household = frame.table("household").copy()
        for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
            household[column] = household_draws[column].to_numpy()
        household["has_fuel_consumption"] = recipient["has_fuel_consumption"].to_numpy(
            dtype=bool
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
                    frame, UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = (
            *imputation.fit_weight_records,
            *incidence.fit_weight_records,
        )
        self.last_result = UKLCFSConsumptionResult(
            frame=result,
            support_clip=clip_result.receipt,
            donor_uprating=uprating_receipt,
            fuel_flag=fuel_flag_receipt,
            bus_pricing=bus_pricing_receipt,
            energy_pricing=None if energy is None else energy.receipt,
            energy_rake=energy_rake_receipt,
            fuel_litres_audit=litres_audit,
            donor_floor=donor_floor_receipt,
            road_fuel_incidence=incidence.receipt,
            road_fuel_level=road_fuel_level_receipt,
            recomposed_totals=recomposed_totals_receipt,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


@dataclass(frozen=True)
class UKLCFSConsumptionImputationResult:
    draws: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]


def fuel_litres_audit(
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    stage: SourceStageSpec,
) -> dict[str, Any] | None:
    """Frame road-fuel litres against HMRC clearances times the OBR cars share.

    Diagnostic only (microcosm#890 C7): for each fuel column the declared
    uprating moves by the vendored litre proxy, the frame's prior-weighted
    spend is divided by the DESNZ pump price of the uprating's target year and
    compared with the HMRC fiscal-year litres scaled by the OBR cars share of
    fuel duty receipts (the household frame carries cars, not lorries or
    vans). Recorded in the stage evidence; nothing is gated on it.
    """

    parameters = uprating_operation(stage)
    if parameters is None:
        return None
    to_period = int(parameters["to_period"])
    weight_values = np.asarray(weights, dtype=float)
    fuels: dict[str, Any] = {}
    resource = None
    for column, spec in parameters["columns"].items():
        if spec.get("basis") != "vendored_litre_proxy" or column not in household_draws:
            continue
        resource = str(spec["resource"])
        price_rows = vendored_rows(
            resource,
            concept=str(spec["price_concept"]),
            period_type="calendar_year",
            period_value=to_period,
        )
        litre_rows = vendored_rows(
            resource,
            concept=UK_HMRC_LITRES_CONCEPTS[str(column)],
            fiscal_start=f"{to_period}-04-01",
        )
        if len(price_rows) != 1 or len(litre_rows) != 1:
            raise ValueError(
                f"{resource}: litres audit needs one price and one litres row."
            )
        price_pence = float(price_rows[0]["value"])
        hmrc_litres = float(litre_rows[0]["value"])
        spend = _numeric(household_draws[column]).to_numpy(dtype=float)
        spend_total = float(np.dot(spend, weight_values))
        frame_litres = spend_total / (price_pence / 100.0)
        fuels[str(column)] = {
            "price_concept": spec["price_concept"],
            "price_pence_per_litre": price_pence,
            "weighted_spend_gbp": spend_total,
            "frame_litres": frame_litres,
            "hmrc_concept": UK_HMRC_LITRES_CONCEPTS[str(column)],
            "hmrc_litres_all_road_users": hmrc_litres,
        }
    if not fuels:
        return None
    assert resource is not None

    def receipts(category: str) -> float:
        rows = vendored_rows(
            resource,
            concept=UK_OBR_FUEL_DUTY_RECEIPTS_CONCEPT,
            fiscal_start=f"{to_period}-04-01",
            dimensions={"vehicle_category": category},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: litres audit needs one OBR {category!r} row."
            )
        return float(rows[0]["value"])

    cars, total = receipts(UK_OBR_CARS_CATEGORY), receipts(UK_OBR_TOTAL_CATEGORY)
    cars_share = cars / total
    for entry in fuels.values():
        entry["cars_litres_benchmark"] = (
            entry["hmrc_litres_all_road_users"] * cars_share
        )
        entry["frame_over_cars_benchmark"] = (
            entry["frame_litres"] / entry["cars_litres_benchmark"]
            if entry["cars_litres_benchmark"] > 0
            else None
        )
    frame_total = sum(entry["frame_litres"] for entry in fuels.values())
    benchmark_total = sum(entry["cars_litres_benchmark"] for entry in fuels.values())
    return {
        "resource": resource,
        "period_value": to_period,
        "fiscal_start": f"{to_period}-04-01",
        "obr_fuel_duty_receipts": {
            "cars_gbp": cars,
            "total_gbp": total,
            "cars_share": cars_share,
        },
        "fuels": fuels,
        "frame_litres_total": frame_total,
        "cars_litres_benchmark_total": benchmark_total,
        "frame_over_cars_benchmark": (
            frame_total / benchmark_total if benchmark_total > 0 else None
        ),
        # The per-fuel ratios apply the all-fuel cars share to each fuel's
        # all-road-user litres, but diesel is mostly vans and lorries, so
        # only the total is a benchmark (microcosm#1113; a per-fuel cars
        # split waits on PolicyEngine/chronicle#322).
        "per_fuel_ratio_basis": UK_LITRES_AUDIT_PER_FUEL_BASIS,
        "gated": False,
    }


@dataclass(frozen=True)
class RoadFuelIncidence:
    """Road-fuel draws after the incidence redraw, its receipt and fit records."""

    draws: pd.DataFrame
    receipt: Mapping[str, Any] | None
    fit_weight_records: tuple[FitWeightRecord, ...] = ()


def road_fuel_incidence_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``redraw_zero_road_fuel`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == REDRAW_ZERO_ROAD_FUEL_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def redraw_zero_road_fuel(
    household_draws: pd.DataFrame,
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    flagged: np.ndarray,
    household_ids: np.ndarray,
    weights: Sequence[float],
    parameters: Mapping[str, Any],
) -> RoadFuelIncidence:
    """Give every household the fuel flag marks a positive road-fuel spend.

    The LCFS diary covers two weeks, so about a third of car-owning donors
    record no fuel purchase and the chain reproduces that zero share among the
    households the fuel flag marks as buying petrol or diesel; over a year
    such a car is refuelled. For a flagged household whose chain draw is zero,
    petrol plus diesel is redrawn from the donor's conditional distribution
    among households with positive road fuel, on the chain's predictors: a
    weighted regime-gated QRF draws the total, positive by construction, then
    the petrol share of it (which keeps diesel-only and mixed households), at
    quantiles keyed on the household id. Positive draws are kept; the level
    step then rescales the total (microcosm#1113).
    """

    from microcosm.fit.qrf import Regime, RegimeGatedQRF

    columns = tuple(str(column) for column in parameters.get("columns", ()))
    if columns != UK_LCFS_ROAD_FUEL_COLUMNS:
        raise ValueError(
            f"{REDRAW_ZERO_ROAD_FUEL_KIND} must redraw {UK_LCFS_ROAD_FUEL_COLUMNS}, "
            f"not {columns}."
        )
    if parameters.get("flag") != "has_fuel_consumption":
        raise ValueError(
            f"{REDRAW_ZERO_ROAD_FUEL_KIND} must redraw the has_fuel_consumption "
            "households."
        )
    rule = str(parameters.get("rule") or "")
    if rule != UK_LCFS_ROAD_FUEL_REDRAW_RULE:
        raise ValueError(f"unsupported {REDRAW_ZERO_ROAD_FUEL_KIND} rule {rule!r}.")
    seed = parameters.get("seed")
    n_estimators = parameters.get("n_estimators")
    salt = parameters.get("salt")
    if not isinstance(seed, int) or not isinstance(n_estimators, int):
        raise ValueError(f"{REDRAW_ZERO_ROAD_FUEL_KIND} needs integer seed and trees.")
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"{REDRAW_ZERO_ROAD_FUEL_KIND} needs a declared salt.")
    flagged = np.asarray(flagged, dtype=bool)
    weight = np.asarray(weights, dtype=float)
    spend = household_draws[list(columns)].to_numpy(dtype=float)
    before = spend.sum(axis=1)
    zero = flagged & ~(before > 0)
    redrawn = household_draws.copy()
    regimes: dict[str, str] | None = None
    training_rows = 0
    records: tuple[FitWeightRecord, ...] = ()
    if zero.any():
        donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
            donor, recipient
        )
        donor_total = donor_encoded["petrol_spending"].to_numpy(
            dtype=float
        ) + donor_encoded["diesel_spending"].to_numpy(dtype=float)
        positive = donor_total > 0
        training_rows = int(positive.sum())
        if training_rows == 0:
            raise ValueError("LCFS donor records no positive road-fuel spend.")
        train = donor_encoded.loc[positive, [*predictors, "household_weight"]]
        train = train.reset_index(drop=True)
        train[UK_LCFS_ROAD_FUEL_TOTAL] = donor_total[positive]
        train[UK_LCFS_PETROL_SHARE] = (
            donor_encoded.loc[positive, "petrol_spending"].to_numpy(dtype=float)
            / donor_total[positive]
        )
        targets = [UK_LCFS_ROAD_FUEL_TOTAL, UK_LCFS_PETROL_SHARE]
        fitted = RegimeGatedQRF(n_estimators=n_estimators, seed=seed).fit(
            train, list(predictors), targets, weights="household_weight"
        )
        regimes = {target: str(regime) for target, regime in fitted.regimes().items()}
        if regimes[UK_LCFS_ROAD_FUEL_TOTAL] != Regime.POSITIVE_ONLY:
            raise ValueError(
                f"the road-fuel total fitted regime {regimes[UK_LCFS_ROAD_FUEL_TOTAL]!r}"
                f", not {Regime.POSITIVE_ONLY!r}."
            )
        ids = np.asarray(household_ids)[zero]
        drawn = fitted.predict_from_uniforms(
            recipient_encoded.loc[zero, list(predictors)].reset_index(drop=True),
            quantiles={
                target: stable_identity_uniforms(
                    ids, seed=seed, salt=f"{salt}:{target}"
                )
                for target in targets
            },
            sign_uniforms={
                target: stable_identity_uniforms(
                    ids, seed=seed, salt=f"{salt}:{target}:sign"
                )
                for target in targets
            },
        )
        total = drawn[UK_LCFS_ROAD_FUEL_TOTAL].to_numpy(dtype=float)
        share = np.clip(drawn[UK_LCFS_PETROL_SHARE].to_numpy(dtype=float), 0.0, 1.0)
        rows = np.flatnonzero(zero)
        redrawn.iloc[rows, redrawn.columns.get_loc("petrol_spending")] = share * total
        redrawn.iloc[rows, redrawn.columns.get_loc("diesel_spending")] = (
            1.0 - share
        ) * total
        records = tuple(
            FitWeightRecord(
                f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", fitted.weight_kind
            )
            for target in targets
        )
    after_spend = redrawn[list(columns)].to_numpy(dtype=float)
    after = after_spend.sum(axis=1)

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        mass = float(weight[mask].sum())
        return float(np.dot(weight[mask], values[mask]) / mass) if mass > 0 else None

    flagged_mass = float(weight[flagged].sum())
    redrawn_total = float(np.dot(weight[zero], after[zero]))
    receipt = {
        "operation": REDRAW_ZERO_ROAD_FUEL_KIND,
        "rule": rule,
        "seed": seed,
        "salt": salt,
        "n_estimators": n_estimators,
        "columns": list(columns),
        "regimes": regimes,
        "training_rows": training_rows,
        "flagged_households": int(flagged.sum()),
        "flagged_zero_before": int(zero.sum()),
        "flagged_zero_share_before": (
            float(weight[zero].sum()) / flagged_mass if flagged_mass > 0 else 0.0
        ),
        "flagged_zero_after": int((flagged & ~(after > 0)).sum()),
        "unflagged_with_fuel": int((~flagged & (after > 0)).sum()),
        "mean_positive_before": weighted_mean(before, flagged & (before > 0)),
        "mean_redrawn": weighted_mean(after, zero),
        "mean_flagged_after": weighted_mean(after, flagged),
        "redrawn_petrol_share": (
            float(np.dot(weight[zero], after_spend[zero, 0])) / redrawn_total
            if redrawn_total > 0
            else None
        ),
        "weighted_total_before": float(np.dot(weight, before)),
        "weighted_total_after": float(np.dot(weight, after)),
    }
    return RoadFuelIncidence(redrawn, receipt, records)


def lcfs_road_fuel_incidence(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    household_ids: np.ndarray,
    weights: Sequence[float],
) -> RoadFuelIncidence:
    """Apply the declared ``redraw_zero_road_fuel`` step (none if undeclared)."""

    parameters = road_fuel_incidence_operation(stage)
    if parameters is None:
        return RoadFuelIncidence(household_draws, None)
    return redraw_zero_road_fuel(
        household_draws,
        donor=donor,
        recipient=recipient,
        flagged=recipient["has_fuel_consumption"].to_numpy(dtype=bool),
        household_ids=household_ids,
        weights=weights,
        parameters=parameters,
    )


@dataclass(frozen=True)
class RoadFuelLevel:
    """The household road-fuel level the ``level_road_fuel`` step scales to.

    ``published`` is the ONS Consumer Trends COICOP 07.2.2 spend (fuels and
    lubricants for personal transport equipment); ``other_fuels_share`` the
    part of it that is neither petrol nor diesel, taken from the LCFS donor's
    own split of 07.2.2. ``level`` is what the frame's petrol plus diesel
    totals at prior weights after the step.
    """

    published: float
    other_fuels_share: float
    receipt: Mapping[str, Any]

    @property
    def level(self) -> float:
        return self.published * (1.0 - self.other_fuels_share)


def road_fuel_level_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``level_road_fuel`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == LEVEL_ROAD_FUEL_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def lcfs_other_road_fuel_share(lcfs_household: pd.DataFrame) -> float:
    """The donor's weighted share of COICOP 07.2.2 that is neither petrol nor diesel."""

    household = _lowercase(lcfs_household)
    _require_columns(
        household, ("weighta", "c72211", "c72212", *UK_LCFS_OTHER_ROAD_FUEL_CODES)
    )
    weight = _numeric(household["weighta"]).to_numpy(dtype=float)
    other = sum(
        _numeric(household[code]).to_numpy(dtype=float)
        for code in UK_LCFS_OTHER_ROAD_FUEL_CODES
    )
    total = (
        _numeric(household["c72211"]).to_numpy(dtype=float)
        + _numeric(household["c72212"]).to_numpy(dtype=float)
        + other
    )
    denominator = float(np.dot(weight, total))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("LCFS donor records no COICOP 07.2.2 road-fuel spend.")
    return float(np.dot(weight, other)) / denominator


def road_fuel_level(
    parameters: Mapping[str, Any], *, other_fuels_share: float
) -> RoadFuelLevel:
    """Resolve the declared ONS road-fuel level from the vendored rows."""

    columns = tuple(str(column) for column in parameters.get("columns", ()))
    if columns != UK_LCFS_ROAD_FUEL_COLUMNS:
        raise ValueError(
            f"{LEVEL_ROAD_FUEL_KIND} must level {UK_LCFS_ROAD_FUEL_COLUMNS}, "
            f"not {columns}."
        )
    resource = str(parameters.get("resource") or "")
    if resource != UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE:
        raise ValueError(
            f"{LEVEL_ROAD_FUEL_KIND} must read {UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE!r}, "
            f"not {resource!r}."
        )
    rule = str(parameters.get("other_fuels_rule") or "")
    if rule != UK_LCFS_ROAD_FUEL_OTHER_SHARE_RULE:
        raise ValueError(
            f"unsupported {LEVEL_ROAD_FUEL_KIND} other_fuels_rule {rule!r}."
        )
    codes = tuple(str(code) for code in parameters.get("other_fuels_codes", ()))
    if codes != UK_LCFS_OTHER_ROAD_FUEL_CODES:
        raise ValueError(
            f"{LEVEL_ROAD_FUEL_KIND} other_fuels_codes must be "
            f"{UK_LCFS_OTHER_ROAD_FUEL_CODES}, not {codes}."
        )
    if str(parameters.get("period_type")) != "calendar_year":
        raise ValueError(f"{LEVEL_ROAD_FUEL_KIND} binds a calendar-year ONS level.")
    if not 0.0 <= other_fuels_share < 1.0:
        raise ValueError(f"other-fuels share {other_fuels_share} is outside [0, 1).")
    period_value = int(parameters["period_value"])
    rows = vendored_rows(
        resource,
        concept=str(parameters["concept"]),
        period_type="calendar_year",
        period_value=period_value,
        geography_id="K02000001",
        dimensions={"coicop": str(parameters["coicop"]), "frequency": "annual"},
    )
    if len(rows) != 1:
        raise ValueError(
            f"{resource}: expected one {parameters['coicop']} row for "
            f"{period_value}, found {len(rows)}."
        )
    published = float(rows[0]["value"])
    if not np.isfinite(published) or published <= 0:
        raise ValueError(f"{resource}: the road-fuel level must be positive.")
    receipt = {
        "operation": LEVEL_ROAD_FUEL_KIND,
        "resource": resource,
        "concept": str(parameters["concept"]),
        "coicop": str(parameters["coicop"]),
        "period_type": "calendar_year",
        "period_value": period_value,
        "source_record_id": str(rows[0].get("source_record_id", "")),
        "published": published,
        "other_fuels_rule": rule,
        "other_fuels_codes": list(UK_LCFS_OTHER_ROAD_FUEL_CODES),
        "other_fuels_share": float(other_fuels_share),
        "level": published * (1.0 - other_fuels_share),
        "columns": list(UK_LCFS_ROAD_FUEL_COLUMNS),
    }
    return RoadFuelLevel(published, float(other_fuels_share), receipt)


def level_road_fuel(
    household_draws: pd.DataFrame,
    *,
    level: RoadFuelLevel,
    weights: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale petrol and diesel by one factor so the prior-weighted total is the level.

    One factor keeps the drawn petrol/diesel mix and every household's
    relative spend; only the level moves (microcosm#1113: the LCFS diary
    records about a quarter less road fuel than ONS household spending).
    """

    columns = list(UK_LCFS_ROAD_FUEL_COLUMNS)
    weight_values = np.asarray(weights, dtype=float)
    spend = household_draws[columns].to_numpy(dtype=float)
    before = float(np.dot(weight_values, spend.sum(axis=1)))
    if not np.isfinite(before) or before <= 0:
        raise ValueError("the frame draws no road-fuel spend to level.")
    factor = level.level / before
    result = household_draws.copy()
    result[columns] = spend * factor
    after = result[columns].to_numpy(dtype=float)
    petrol = float(np.dot(weight_values, after[:, 0]))
    receipt = {
        **dict(level.receipt),
        "frame_before": before,
        "factor": float(factor),
        "frame_after": float(np.dot(weight_values, after.sum(axis=1))),
        "petrol_share_of_level": petrol / level.level,
    }
    return result, receipt


def lcfs_road_fuel_level(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    lcfs_household: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply the declared ``level_road_fuel`` step (none if undeclared)."""

    parameters = road_fuel_level_operation(stage)
    if parameters is None:
        return household_draws, None
    level = road_fuel_level(
        parameters, other_fuels_share=lcfs_other_road_fuel_share(lcfs_household)
    )
    return level_road_fuel(household_draws, level=level, weights=weights)


def recompose_from_remainder_operation(
    stage: SourceStageSpec,
) -> Mapping[str, Any] | None:
    """The stage's declared ``recompose_from_remainder`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == RECOMPOSE_FROM_REMAINDER_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def check_recomposed_parents(parameters: Mapping[str, Any]) -> None:
    """Refuse a declaration that differs from the totals the stage recomposes."""

    parents = parameters.get("parents")
    if not isinstance(parents, Mapping) or set(parents) != set(
        UK_LCFS_RECOMPOSED_PARENTS
    ):
        raise ValueError(
            f"{RECOMPOSE_FROM_REMAINDER_KIND} must declare "
            f"{sorted(UK_LCFS_RECOMPOSED_PARENTS)}."
        )
    for parent, (subtracted, components) in UK_LCFS_RECOMPOSED_PARENTS.items():
        declared = parents[parent]
        if (
            not isinstance(declared, Mapping)
            or tuple(declared.get("drawn_subtracts", ())) != subtracted
            or tuple(declared.get("components", ())) != components
        ):
            raise ValueError(
                f"{RECOMPOSE_FROM_REMAINDER_KIND} declares {parent!r} as "
                f"{declared!r}, not its draw less {subtracted} plus {components}."
            )


def uncarried_spend(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The published spend inside a recomposed total that no column carries.

    ONS Consumer Trends classes read from the vendored rows the declaration
    names: the liquid and solid fuels of COICOP 04.5, which have no column of
    their own and stay inside the housing remainder at the diary's level.
    """

    spec = parameters.get("uncarried")
    if not isinstance(spec, Mapping):
        raise ValueError(f"{RECOMPOSE_FROM_REMAINDER_KIND} must declare uncarried.")
    resource = str(spec.get("resource") or "")
    if resource not in UK_LCFS_VENDORED_RESOURCES:
        raise ValueError(f"{resource!r} is not a vendored lcfs_consumption resource.")
    period_value = int(spec["period_value"])
    classes = []
    for entry in spec.get("classes", ()):
        coicop, concept = str(entry["coicop"]), str(entry["concept"])
        rows = vendored_rows(
            resource,
            concept=concept,
            period_type="calendar_year",
            period_value=period_value,
            geography_id="K02000001",
            dimensions={"coicop": coicop, "frequency": "annual"},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: expected one {coicop} row for {period_value}, "
                f"found {len(rows)}."
            )
        classes.append(
            {
                "coicop": coicop,
                "concept": concept,
                "source_record_id": str(rows[0].get("source_record_id", "")),
                "value": float(rows[0]["value"]),
            }
        )
    if not classes:
        raise ValueError(f"{RECOMPOSE_FROM_REMAINDER_KIND} declares no classes.")
    return {
        "resource": resource,
        "period_value": period_value,
        "classes": classes,
        "total": float(sum(entry["value"] for entry in classes)),
    }


def recompose_parent_totals(
    household_draws: pd.DataFrame,
    *,
    drawn: pd.DataFrame,
    parameters: Mapping[str, Any],
    weights: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write each recomposed total around the levelled parts the stage set.

    A total keeps its own chain draw less the chain's draws of the parts the
    stage re-levels (``drawn``, kept before any step touched them; floored at
    zero where the parts outgrow the total), and adds the levelled parts back,
    so it always contains the electricity, gas, petrol and diesel at their
    published scale (microcosm#1113). The chain draws each part after, and
    conditional on, its total, so the remainder is the chain's own split; a
    remainder drawn as a chain target of its own over-drew the vehicle-purchase
    tail of transport (10 to 13 percent above the donor in-sample).
    """

    check_recomposed_parents(parameters)
    weight = np.asarray(weights, dtype=float)
    result = household_draws.copy()
    parents: dict[str, Any] = {}
    for parent, (subtracted, components) in UK_LCFS_RECOMPOSED_PARENTS.items():
        drawn_total = result[parent].to_numpy(dtype=float)
        drawn_parts = sum(drawn[column].to_numpy(dtype=float) for column in subtracted)
        rest = np.maximum(drawn_total - drawn_parts, 0.0)
        parts = sum(result[column].to_numpy(dtype=float) for column in components)
        total = rest + parts
        result[parent] = total
        parents[parent] = {
            "drawn_subtracts": list(subtracted),
            "components": list(components),
            "weighted_drawn_total": float(np.dot(weight, drawn_total)),
            "weighted_drawn_subtracts": float(np.dot(weight, drawn_parts)),
            "rows_floored": int((drawn_parts > drawn_total).sum()),
            "weighted_floored_mass": float(
                np.dot(weight, np.maximum(drawn_parts - drawn_total, 0.0))
            ),
            "weighted_remainder": float(np.dot(weight, rest)),
            "weighted_components": float(np.dot(weight, parts)),
            "weighted_total": float(np.dot(weight, total)),
            "minimum_remainder": float(rest.min()) if rest.size else 0.0,
            "rows_below_components": int((total < parts).sum()),
        }
    receipt = {
        "operation": RECOMPOSE_FROM_REMAINDER_KIND,
        "parents": parents,
        "uncarried": uncarried_spend(parameters),
    }
    return result, receipt


def lcfs_recompose_from_remainder(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    drawn: pd.DataFrame,
    weights: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply the declared ``recompose_from_remainder`` step (none if undeclared)."""

    parameters = recompose_from_remainder_operation(stage)
    if parameters is None:
        return household_draws, None
    return recompose_parent_totals(
        household_draws, drawn=drawn, parameters=parameters, weights=weights
    )


@dataclass(frozen=True)
class LCFSEnergyPricing:
    """The declared prices, NEED margins, gas connection and level, with receipts."""

    prices: EnergyPrices
    margins: NeedMargins
    gas_connected: str
    connection_shares: Mapping[str, float | None] | None
    disconnect_rule: str
    level: Mapping[str, float]
    receipt: dict[str, Any]
    disconnect_seed: int = 0


def lcfs_energy_pricing(stage: SourceStageSpec) -> LCFSEnergyPricing | None:
    """Resolve the declared ``price_domestic_energy`` operation (none if undeclared)."""

    parameters = pricing_operation(stage)
    if parameters is None:
        return None
    columns = dict(parameters["columns"])
    if columns != {"electricity": "electricity_consumption", "gas": "gas_consumption"}:
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} must price the two energy spend columns."
        )
    margins_resource = str(
        parameters.get("margins_resource") or UK_NEED_ENERGY_FACTS_RESOURCE
    )
    if "margins_period_value" not in parameters:
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} declares no margins_period_value (the "
            "NEED consumption year the shape comes from)."
        )
    prices, prices_receipt = qep_prices(parameters)
    margins = need_margins_from_facts(
        margins_resource, period_value=int(parameters["margins_period_value"])
    )
    level, level_receipt = published_energy_level(parameters)
    gas_connected = str(parameters.get("gas_connected") or "")
    if gas_connected == GAS_CONNECTED_PUBLISHED_METER_SHARE:
        shares, connection_receipt = published_gas_connected_shares(parameters)
    elif gas_connected == GAS_CONNECTED_POSITIVE_SPEND:
        shares, connection_receipt = None, {"rule": GAS_CONNECTED_POSITIVE_SPEND}
    else:
        raise ValueError(
            f"gas_connected must be {GAS_CONNECTED_POSITIVE_SPEND!r} or "
            f"{GAS_CONNECTED_PUBLISHED_METER_SHARE!r}, not {gas_connected!r}."
        )
    disconnect_rule = str(
        parameters.get("disconnect_rule") or DISCONNECT_LOWEST_DRAWN_GAS_FIRST
    )
    disconnect_seed = parameters.get("seed", 0)
    if disconnect_rule == DISCONNECT_IDENTITY_UNIFORM_ORDER and not (
        isinstance(disconnect_seed, int) and disconnect_seed >= 0
    ):
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} with disconnect_rule "
            f"{DISCONNECT_IDENTITY_UNIFORM_ORDER!r} must declare a non-negative "
            "integer seed."
        )
    return LCFSEnergyPricing(
        prices=prices,
        margins=margins,
        gas_connected=gas_connected,
        connection_shares=shares,
        disconnect_rule=disconnect_rule,
        disconnect_seed=int(disconnect_seed),
        level=level,
        receipt={
            **prices_receipt,
            "need_margins": margins.receipt,
            "level": level_receipt,
            "gas_connection": connection_receipt,
        },
    )


def energy_spend_to_kwh(
    table: pd.DataFrame, *, energy: LCFSEnergyPricing, region: np.ndarray
) -> pd.DataFrame:
    """Add ``electricity_kwh`` and ``gas_kwh`` at the declared regional average prices."""

    result = table.copy()
    electricity = _numeric(result["electricity_consumption"]).to_numpy(dtype=float)
    gas = _numeric(result["gas_consumption"]).to_numpy(dtype=float)
    result[ELECTRICITY_KWH] = spend_to_kwh(
        electricity, frs_region=region, fuel="electricity", prices=energy.prices
    )
    result[GAS_KWH] = spend_to_kwh(
        gas, frs_region=region, fuel="gas", prices=energy.prices, connected=gas > 0
    )
    return result


def energy_kwh_to_spend(
    table: pd.DataFrame, *, energy: LCFSEnergyPricing, region: np.ndarray
) -> pd.DataFrame:
    """Price the kWh columns back to spend (gas fixed cost where connected) and drop them."""

    result = table.copy()
    gas_kwh = result[GAS_KWH].to_numpy(dtype=float)
    result["electricity_consumption"] = kwh_to_spend(
        result[ELECTRICITY_KWH].to_numpy(dtype=float),
        frs_region=region,
        fuel="electricity",
        prices=energy.prices,
    )
    result["gas_consumption"] = kwh_to_spend(
        gas_kwh,
        frs_region=region,
        fuel="gas",
        prices=energy.prices,
        connected=gas_kwh > 0,
    )
    return result.drop(columns=[ELECTRICITY_KWH, GAS_KWH])


def rake_recipient_energy(
    household_draws: pd.DataFrame,
    *,
    energy: LCFSEnergyPricing,
    region: np.ndarray,
    income: np.ndarray,
    tenure: np.ndarray,
    accommodation: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    identity: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Price the drawn spend to kWh, impose the published gas connection, rake the
    NEED shape at the DESNZ level, price back.

    ``identity`` (the household ids, row-aligned) keys the disconnection walk
    when the declared rule is ``identity_uniform_order``."""

    in_kwh = energy_spend_to_kwh(household_draws, energy=energy, region=region)
    drawn_gas = in_kwh[GAS_KWH].to_numpy(dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if energy.gas_connected == GAS_CONNECTED_PUBLISHED_METER_SHARE:
        connected, connection_receipt = impose_gas_connection(
            drawn_gas,
            frs_region=region,
            weights=weight_values,
            shares=energy.connection_shares or {},
            disconnect_rule=energy.disconnect_rule,
            identity=identity,
            seed=energy.disconnect_seed,
        )
        in_kwh.loc[~connected, GAS_KWH] = 0.0
    else:
        connected = drawn_gas > 0
        share = float(np.dot(connected, weight_values)) / max(
            float(weight_values.sum()), 1e-12
        )
        connection_receipt = {"share_before": share, "share_after": share}
    raked, receipt = rake_energy_kwh(
        in_kwh,
        margins=energy.margins,
        frs_region=region,
        income=income,
        weights=weight_values,
        iterations=iterations,
        tenure=tenure,
        accommodation=accommodation,
        use_region_margin=True,
        gas_connected=connected,
        level=energy.level,
    )
    receipt["gas_connection"] = {"rule": energy.gas_connected, **connection_receipt}
    receipt["gas_connected_share"] = float(
        np.dot(raked[GAS_KWH].to_numpy(dtype=float) > 0, weight_values)
        / max(float(weight_values.sum()), 1e-12)
    )
    return energy_kwh_to_spend(raked, energy=energy, region=region), receipt


def _energy_rake_operation(stage: SourceStageSpec, *, margin: str):
    """The declared energy IPF whose margins name ``margin`` (never positional)."""

    for operation in stage.operations:
        if operation.kind == "iterative_proportional_fit" and (
            "electricity_consumption" in operation.parameters.get("columns", ())
            and margin in [str(m) for m in operation.parameters.get("margins", ())]
        ):
            return operation
    return None


def _donor_energy_rake_iterations(stage: SourceStageSpec) -> int:
    """The declared donor-side energy IPF (the one on the gross income band)."""

    operation = _energy_rake_operation(stage, margin="gross_income_band")
    return 1 if operation is None else int(operation.parameters.get("iterations", 1))


def _recipient_energy_rake_iterations(stage: SourceStageSpec) -> int:
    """The declared post-imputation energy IPF (the one with the region margin).

    Refuses a stage without that IPF or its declared sweep count instead of
    falling back to a default: the energy_rake gate reads the residual's
    convergence over the last sweeps of exactly the declared count.
    """

    operation = _energy_rake_operation(stage, margin="region")
    if operation is None or "iterations" not in operation.parameters:
        raise ValueError(
            f"{stage.stage}: the post-imputation energy IPF (the one with the "
            "region margin) and its declared iterations are required."
        )
    return int(operation.parameters["iterations"])


def support_clip_exempt(stage: SourceStageSpec) -> set[str]:
    """Columns the declared ``support_clip`` exempts (raked columns keep their level)."""

    for operation in stage.operations:
        if operation.kind == "support_clip":
            declared = operation.parameters.get("exempt")
            if declared:
                return {str(column) for column in declared}
    return set(UK_LCFS_DEFAULT_SUPPORT_CLIP_EXEMPT)


def bus_pricing_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``price_bus_journeys`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == PRICE_BUS_JOURNEYS_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def lcfs_bus_fare_pricing(
    stage: SourceStageSpec, household_draws: pd.DataFrame, frame: Frame
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Replace the priced regions' bus fares with journeys times the published yield.

    The chain's raw draw (clipped to donor support) stands where no area
    prices (Wales); everywhere else the emitted value is the sum over the
    household's non-eligible persons of their ``nts_bus_travel`` journeys
    times boardings per resident trip times the yield per fare-paying
    boarding (microcosm#930). The chain conditioned later targets on the raw
    draw, which the receipt states.
    """

    parameters = bus_pricing_operation(stage)
    if parameters is None:
        return household_draws, None
    column = str(parameters["column"])
    if column not in household_draws:
        raise ValueError(
            f"price_bus_journeys names {column!r}, which the chain did not draw."
        )
    prices = bus_fare_prices(parameters, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    trips_columns = {
        str(k): str(v) for k, v in dict(parameters["trips_columns"]).items()
    }
    household = frame.table("household")
    fares, priced, receipt = price_bus_journeys(
        frame.table("person"),
        household,
        prices=prices,
        trips_columns=trips_columns,
        eligibility_column=str(parameters["eligibility_column"]),
        household_weights=frame.weights_for("household").values,
        raw_household_fares=household_draws[column].to_numpy(dtype=float),
    )
    priced_draws = household_draws.copy()
    priced_draws[column] = fares
    return priced_draws, {
        "operation": PRICE_BUS_JOURNEYS_KIND,
        "column": column,
        "chain_conditioned_on": "raw_draw",
        "households_priced": int(priced.sum()),
        **prices.receipt,
        **receipt,
    }


def lcfs_donor_uprating(
    stage: SourceStageSpec,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """The stage's declared donor uprating factors and receipt (none if undeclared)."""

    parameters = uprating_operation(stage)
    if parameters is None:
        return {}, None
    declared = {
        str(spec.get("resource"))
        for spec in parameters["columns"].values()
        if spec.get("resource")
    }
    if declared - {UK_LCFS_ROAD_FUEL_RESOURCE}:
        raise ValueError(
            "lcfs_consumption uprating may only read "
            f"{UK_LCFS_ROAD_FUEL_RESOURCE!r}; declared {sorted(declared)}."
        )
    return donor_uprating_factors(parameters)


def lcfs_ice_share(stage: SourceStageSpec) -> tuple[float, dict[str, Any]]:
    """The declared fuel-buyer share of car households and its receipt.

    Read from the stage's ``assign_binary_from_rate`` declaration for
    ``has_fuel_consumption``: the vendored DfT VEH1103 licensed-car stock at
    the declared period and geography, one minus the zero-emission share.
    """

    parameters = _operation_parameters(
        stage, "assign_binary_from_rate", target="has_fuel_consumption"
    )
    return ice_share_from_licensed_cars(parameters)


def ice_share_from_licensed_cars(
    parameters: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """1 - (zero-emission licensed cars / all licensed cars) from vendored rows."""

    resource = str(parameters.get("rate_resource") or "")
    if resource != UK_LCFS_LICENSED_CARS_RESOURCE:
        raise ValueError(
            "has_fuel_consumption rate must come from "
            f"{UK_LCFS_LICENSED_CARS_RESOURCE!r}, not {resource!r}."
        )
    rule = str(parameters.get("rate_rule") or "")
    if rule != UK_LCFS_ICE_SHARE_RULE:
        raise ValueError(f"unsupported has_fuel_consumption rate_rule {rule!r}.")
    period_value = int(parameters["period_value"])
    geography_id = str(parameters["geography_id"])
    zero_emission_types = tuple(
        str(fuel_type) for fuel_type in parameters.get("zero_emission_fuel_types", ())
    )
    if not zero_emission_types:
        raise ValueError("has_fuel_consumption declares no zero_emission_fuel_types.")

    def licensed(fuel_type: str) -> tuple[float, str]:
        rows = vendored_rows(
            resource,
            period_type="calendar_year",
            period_value=period_value,
            geography_id=geography_id,
            dimensions={"fuel_type": fuel_type},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: expected one {fuel_type!r} row for {period_value} "
                f"{geography_id}, found {len(rows)}."
            )
        return float(rows[0]["value"]), str(rows[0].get("source_record_id", ""))

    all_cars, all_record = licensed("all")
    zero_emission = {
        fuel_type: licensed(fuel_type) for fuel_type in zero_emission_types
    }
    if not np.isfinite(all_cars) or all_cars <= 0:
        raise ValueError(f"{resource}: licensed-car total must be positive.")
    zero_total = sum(count for count, _ in zero_emission.values())
    rate = 1.0 - zero_total / all_cars
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"{resource}: fuel-buyer share {rate} is outside (0, 1].")
    receipt = {
        "resource": resource,
        "rule": rule,
        "period_value": period_value,
        "geography_id": geography_id,
        "licensed_cars": all_cars,
        "zero_emission_licensed_cars": {
            fuel_type: count for fuel_type, (count, _) in zero_emission.items()
        },
        "rate": float(rate),
        "source_record_ids": [
            all_record,
            *(record for _, record in zero_emission.values()),
        ],
    }
    return float(rate), receipt


def vehicle_count(values: pd.Series) -> pd.Series:
    """Round and clip a vehicle count to ``UK_LCFS_VEHICLE_COUNT_RANGE``."""

    low, high = UK_LCFS_VEHICLE_COUNT_RANGE
    return np.rint(_numeric(values)).clip(low, high).astype("int64")


def fuel_flag_evidence(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    recipient_weights: Sequence[float],
    ice_share_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Weighted fuel-household shares on both sides of the imputation (donor design weights, frame prior weights)."""

    donor_weights = _numeric(donor["household_weight"]).to_numpy(dtype=float)
    donor_vehicles = _numeric(donor["num_vehicles"]).to_numpy(dtype=float) > 0
    donor_fuel = (
        _numeric(donor["petrol_spending"]) + _numeric(donor["diesel_spending"])
    ).to_numpy(dtype=float) > 0
    weights = np.asarray(recipient_weights, dtype=float)
    recipient_vehicles = _numeric(recipient["num_vehicles"]).to_numpy(dtype=float) > 0
    flagged = recipient["has_fuel_consumption"].to_numpy(dtype=bool)
    return {
        "ice_share": dict(ice_share_receipt),
        "donor": {
            "households": int(len(donor)),
            "with_vehicles_share": _weighted_share(donor_vehicles, donor_weights),
            "positive_fuel_share": _weighted_share(donor_fuel, donor_weights),
            "positive_fuel_share_among_vehicle_households": _weighted_share(
                donor_fuel[donor_vehicles], donor_weights[donor_vehicles]
            ),
        },
        "recipient": {
            "households": int(len(recipient)),
            "with_vehicles_share": _weighted_share(recipient_vehicles, weights),
            "flagged_share": _weighted_share(flagged, weights),
        },
    }


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return float(np.sum(weights[np.asarray(mask, dtype=bool)]) / total)


def clean_lcfs_consumption_table(
    lcfs_person: pd.DataFrame,
    lcfs_household: pd.DataFrame,
    *,
    uprating: Mapping[str, float] | None = None,
    energy: LCFSEnergyPricing | None = None,
    donor_rake_iterations: int = 1,
) -> pd.DataFrame:
    """Return the LCFS donor table with annualized consumption variables.

    ``uprating`` maps donor columns to the declared factors that move the
    2023-24 diary to the FRS 2024-25 base year (``uprate_donor_columns``);
    it is applied after annualisation and before the donor-side NEED rake, so
    the income bands and the support-clip ranges see base-year values.
    ``energy`` (the declared ``price_domestic_energy``) converts the diary's
    energy spend to kWh at the FY2024-25 regional average prices paid, rakes
    the income margin to the NEED means (shape only, no level) and prices back; without it the energy columns
    are the raw diary spend.
    """

    person = _lowercase(lcfs_person).rename(columns=PERSON_LCFS_RENAMES)
    household = _lowercase(lcfs_household).rename(columns=HOUSEHOLD_LCFS_RENAMES)
    _require_columns(household, ("case", *HOUSEHOLD_LCFS_RENAMES.values()))
    _require_columns(person, ("case", *PERSON_LCFS_RENAMES.values()))
    household["region"] = _numeric(household["region"]).map(LCFS_REGIONS)
    household["num_vehicles"] = vehicle_count(household["num_vehicles"])
    household["tenure_type"] = _numeric(_lowercase(lcfs_household)["a122"]).map(
        LCFS_TENURE_MAP
    )
    household["accommodation_type"] = _numeric(_lowercase(lcfs_household)["a121"]).map(
        LCFS_ACCOMM_MAP
    )
    household = derive_energy_from_lcfs(household)
    household = household.rename(columns=CONSUMPTION_VARIABLE_RENAMES)
    for code in BUS_FARE_LCFS_CODES:
        if code not in household:
            raise ValueError(f"LCFS household donor is missing {code!r}.")
    household["bus_fare_spending"] = sum(
        _numeric(household[code]) for code in BUS_FARE_LCFS_CODES
    )
    annualize = [
        *CONSUMPTION_VARIABLE_RENAMES.values(),
        "bus_fare_spending",
        "hbai_household_net_income",
        "household_gross_income",
        "electricity_consumption",
        "gas_consumption",
    ]
    for column in annualize:
        household[column] = _numeric(household[column]) * WEEKS_IN_YEAR
    for column in PERSON_LCFS_RENAMES.values():
        totals = person.groupby("case")[column].sum()
        household[column] = household["case"].map(totals).fillna(0.0) * WEEKS_IN_YEAR
    household["household_weight"] = _numeric(household["household_weight"]) * 1_000
    if uprating:
        household = apply_donor_uprating(household, uprating)
    if energy is not None:
        region = household["region"].astype(str).to_numpy()
        in_kwh = energy_spend_to_kwh(household, energy=energy, region=region)
        raked, _receipt = rake_energy_kwh(
            in_kwh,
            margins=energy.margins,
            frs_region=region,
            income=household["household_gross_income"].to_numpy(dtype=float),
            weights=None,
            iterations=donor_rake_iterations,
            gas_connected=in_kwh[GAS_KWH].to_numpy(dtype=float) > 0,
        )
        household = energy_kwh_to_spend(raked, energy=energy, region=region)
    household["domestic_energy_consumption"] = (
        household["electricity_consumption"] + household["gas_consumption"]
    )
    return household[
        [
            *UK_LCFS_CONSUMPTION_PREDICTORS,
            *UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
            "household_gross_income",
            "household_weight",
        ]
    ].dropna()


def _donor_consumption_floor(stage: SourceStageSpec) -> float:
    """The declared floor of the donor's diary consumption columns."""

    parameters = _operation_parameters(stage, "derive")
    floor = parameters.get("floor")
    if floor is None:
        raise ValueError(
            f"{stage.stage}: the derive operation must declare 'floor', the "
            "value negative diary consumption is raised to on the donor."
        )
    value = float(floor)
    if not math.isfinite(value) or value != 0.0:
        raise ValueError(
            f"{stage.stage}: the declared donor consumption floor must be 0, "
            f"got {floor!r}."
        )
    return value


def floor_negative_donor_consumption(
    donor: pd.DataFrame, *, floor: float = 0.0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Raise negative diary consumption on the donor to ``floor`` (microcosm#1063 c9).

    An LCFS diary spend nets refunds against purchases, so a handful of donor
    rows carry a negative annual total (250 release rows inherited a minimum
    of -14,165 in ``housing_water_and_electricity_consumption`` on the
    2026-09-30 build, because the support clip takes its floor from the
    donor's realised range). The release surface declares these columns
    non-negative, so the donor is floored before the imputation, the clip
    ranges and the rake see it. The receipt records, per consumption column,
    the rows raised and the (negative) mass they carried; columns without a
    negative row are listed with zeros so the receipt names the whole surface.
    """

    floored = donor.copy()
    columns: dict[str, dict[str, float | int]] = {}
    for column in CONSUMPTION_VARIABLE_RENAMES.values():
        if column not in floored.columns:
            raise ValueError(f"LCFS donor is missing consumption column {column!r}.")
        values = _numeric(floored[column]).to_numpy(dtype=float)
        negative = np.isfinite(values) & (values < floor)
        columns[column] = {
            "rows_raised": int(negative.sum()),
            "negative_mass": float(values[negative].sum()),
            "minimum_before": float(np.nanmin(values)) if values.size else 0.0,
        }
        if negative.any():
            floored[column] = np.where(negative, floor, values)
    receipt = {
        "floor": float(floor),
        "columns": columns,
        "rows_raised": int(sum(entry["rows_raised"] for entry in columns.values())),
        "remaining_negative_rows": int(
            sum(
                int((_numeric(floored[column]).to_numpy(dtype=float) < floor).sum())
                for column in columns
            )
        ),
    }
    return floored, receipt


def derive_energy_from_lcfs(household: pd.DataFrame) -> pd.DataFrame:
    """Split LCFS domestic energy into electricity and gas weekly amounts."""

    for column in ("p537", "b226", "b489", "b490"):
        if column not in household:
            raise ValueError(f"LCFS household donor is missing {column!r}.")
    p537 = _numeric(household["p537"])
    b226 = _numeric(household["b226"])
    b489 = _numeric(household["b489"])
    b490 = _numeric(household["b490"])
    dd_mask = (b226 > 0) & (p537 > 0)
    mean_elec_share = (b226[dd_mask] / p537[dd_mask]).clip(0, 1).mean()
    if np.isnan(mean_elec_share):
        mean_elec_share = 0.52
    electricity = np.zeros(len(household))
    gas = np.zeros(len(household))
    mask1 = b226 > 0
    electricity[mask1] = b226[mask1]
    gas[mask1] = np.maximum(p537[mask1] - b226[mask1], 0)
    mask2 = (~mask1) & (b489 > 0) & (b490 > 0)
    electricity[mask2] = np.maximum(b489[mask2] - b490[mask2], 0)
    gas[mask2] = b490[mask2]
    mask3 = (~mask1) & (b489 > 0) & (b490 == 0)
    electricity[mask3] = b489[mask3] * mean_elec_share
    gas[mask3] = b489[mask3] * (1 - mean_elec_share)
    mask4 = (~mask1) & (b489 == 0)
    electricity[mask4] = p537[mask4] * mean_elec_share
    gas[mask4] = p537[mask4] * (1 - mean_elec_share)
    result = household.copy()
    result["electricity_consumption"] = np.maximum(electricity, 0.0)
    result["gas_consumption"] = np.maximum(gas, 0.0)
    return result


def assign_recipient_has_fuel(frame: Frame, *, rate: float, seed: int) -> np.ndarray:
    household = frame.table("household")
    if "num_vehicles" not in household:
        raise KeyError("recipient household table is missing 'num_vehicles'.")
    draws = stable_identity_uniforms(
        household["household_id"].to_numpy(),
        seed=seed,
        salt="lcfs_has_fuel_consumption",
    )
    return (_numeric(household["num_vehicles"]) > 0) & assign_binary_from_rate(
        draws, rate
    )


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize LCFS recipient predictors at household grain."""

    materialized = engine.materialize(
        frame, UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    result = pd.DataFrame(index=household.index)
    for predictor in UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS:
        declared = str(engine.variable_metadata(predictor).entity)
        values = np.asarray(materialized[predictor])
        if declared == "household":
            result[predictor] = values
        elif declared == "person":
            summed = (
                pd.Series(values.astype(float))
                .groupby(person["person_household_id"].to_numpy())
                .sum()
            )
            result[predictor] = (
                summed.reindex(household["household_id"]).fillna(0.0).to_numpy()
            )
        else:
            raise ValueError(f"unsupported LCFS predictor entity {declared!r}.")
    age = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float)
    adult = age >= UK_LCFS_HOUSEHOLD_COUNT_ADULT_AGE
    for predictor, members in zip(
        UK_LCFS_HOUSEHOLD_COUNT_PREDICTORS, (adult, ~adult), strict=True
    ):
        counted = (
            pd.Series(members.astype(float))
            .groupby(person["person_household_id"].to_numpy())
            .sum()
        )
        result[predictor] = (
            counted.reindex(household["household_id"]).fillna(0.0).to_numpy()
        )
    for predictor in ("region", "tenure_type", "accommodation_type"):
        if predictor in household:
            result[predictor] = household[predictor].map(_enum_name).to_numpy()
    if "num_vehicles" not in household:
        raise KeyError(
            "recipient household table is missing 'num_vehicles' "
            "(the was_wealth draw the consumption QRF conditions on)."
        )
    result["num_vehicles"] = vehicle_count(household["num_vehicles"]).to_numpy()
    if "household_gross_income" in household:
        result["household_gross_income"] = household[
            "household_gross_income"
        ].to_numpy()
    else:
        result["household_gross_income"] = result["hbai_household_net_income"]
    return result


def impute_lcfs_consumption(
    donor: pd.DataFrame,
    recipient_predictor_frame: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
) -> UKLCFSConsumptionImputationResult:
    """Chain-draw the targets in the declared order.

    ``bus_fare_spending`` stays a chain target so later targets condition on
    the same raw draw; the emitted fares are then replaced by the priced
    value where an area prices (microcosm#930).
    """

    from microcosm.fit import RegimeGatedQRF

    donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
        donor, recipient_predictor_frame
    )
    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    state = model.start_chain(
        donor_encoded,
        list(predictors),
        list(UK_LCFS_CONSUMPTION_TARGET_COLUMNS),
        weights="household_weight",
    )
    raw = pd.DataFrame(index=recipient_encoded.index)
    draws = pd.DataFrame(index=recipient_encoded.index)
    fit_records: list[FitWeightRecord] = []
    for target in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        features = recipient_encoded.loc[:, list(predictors)]
        result = model.fit_draw_next(
            donor_encoded,
            features,
            raw,
            state=state,
            weights="household_weight",
        )
        raw[target] = result.raw_draw
        draws[target] = result.raw_draw
        fit_records.append(
            FitWeightRecord(
                f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", result.weight_kind
            )
        )
        state = result.state
    return UKLCFSConsumptionImputationResult(draws, tuple(fit_records))


def support_clip_to_donor(
    draws: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    exempt: set[str] | None = None,
) -> UKSupportClipResult:
    return support_clip_to_donor_with_receipt(
        draws,
        donor,
        columns=UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
        stage=UK_LCFS_CONSUMPTION_STAGE_NAME,
        exempt=exempt,
    )


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Donor support per clipped column.

    Levelled columns carry no bounds, nor do the totals written back around
    their levelled components, whose coherence the stage's recomposed-totals
    check holds instead (microcosm#1113).
    """

    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        if column in UK_LCFS_RAKED_COLUMNS or column in UK_LCFS_RECOMPOSED_PARENTS:
            continue
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


def _encode_consumption_predictors(
    donor: pd.DataFrame, recipient: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    categorical = ("region", "tenure_type", "accommodation_type")
    base_predictors = tuple(
        p for p in UK_LCFS_CONSUMPTION_PREDICTORS if p not in categorical
    )
    donor_work = donor.copy()
    recipient_work = recipient.copy()
    combined = pd.concat(
        [
            donor_work.loc[:, categorical].reset_index(drop=True),
            recipient_work.loc[:, categorical].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(
        combined.astype(str), columns=list(categorical), dtype=float
    )
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)

    def encode(table: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
        encoded = table.drop(columns=list(categorical), errors="ignore").copy()
        for column in encoded.columns:
            if column == "household_weight":
                continue
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(
                0.0
            )
        block = block.copy()
        block.index = encoded.index
        return pd.concat([encoded, block], axis=1)

    donor_encoded = encode(donor_work, dummies.iloc[: len(donor_work)])
    recipient_encoded = encode(recipient_work, dummies.iloc[len(donor_work) :])
    return donor_encoded, recipient_encoded, (*base_predictors, *tuple(dummies.columns))


def _lowercase(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    return result


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0)


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"LCFS donor is missing required column(s): {missing}.")


def _artifact(stage: SourceStageSpec, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise ValueError(f"{stage.stage} declares no {role!r} artifact.")


def _operation_parameters(
    stage: SourceStageSpec, kind: str, **match: object
) -> Mapping[str, Any]:
    for operation in stage.operations:
        if operation.kind == kind and all(
            operation.parameters.get(key) == value for key, value in match.items()
        ):
            return dict(operation.parameters)
    raise ValueError(f"{stage.stage} declares no {kind!r} operation for {match}.")


def _operation_seed(stage: SourceStageSpec, kind: str) -> int:
    for operation in stage.operations:
        if operation.kind == kind and isinstance(operation.parameters.get("seed"), int):
            return int(operation.parameters["seed"])
    return 0


def _qrf_n_estimators(stage: SourceStageSpec) -> int:
    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf_chain":
            value = operation.parameters.get("n_estimators", 100)
            if isinstance(value, int) and value > 0:
                return value
    return 100


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("LCFS consumption stage requires caller-supplied donor paths.")
    return Path(path).expanduser().resolve()


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
