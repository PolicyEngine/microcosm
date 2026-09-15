"""UK LCFS consumption imputation stage."""

from __future__ import annotations

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
from microcosm.build.uk_runtime.bus_use_incidence import (
    BusUseIncidenceResult,
    assign_bus_use_incidence,
    incidence_operation,
    nts_band_shares,
)
from microcosm.build.uk_runtime.donor_uprating import (
    apply_donor_uprating,
    donor_uprating_factors,
    uprating_operation,
)
from microcosm.build.uk_runtime.energy_pricing import (
    ELECTRICITY_KWH,
    GAS_KWH,
    UK_NEED_ENERGY_FACTS_RESOURCE,
    UK_OFGEM_PRICE_CAP_RESOURCE,
    EnergyCapRates,
    NeedMargins,
    cap_rates,
    kwh_to_spend,
    need_margins_from_facts,
    pricing_operation,
    rake_energy_kwh,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.fact_raking import (
    rake_operations,
    rake_to_facts,
    resolve_cells,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
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
    "g018": "is_adult",
    "g019": "is_child",
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
UK_LCFS_NTS_BUS_USE_RESOURCE = "nts_bus_use_frequency.json"
UK_LCFS_DFT_BUS_VALUE_RESOURCE = "dft_bus_value_anchors.json"
UK_LCFS_DEVOLVED_BUS_FINANCE_RESOURCE = "devolved_bus_finance.json"
UK_LCFS_VENDORED_RESOURCES = (
    UK_LCFS_ROAD_FUEL_RESOURCE,
    UK_LCFS_LICENSED_CARS_RESOURCE,
    UK_LCFS_NTS_BUS_USE_RESOURCE,
    UK_LCFS_DFT_BUS_VALUE_RESOURCE,
    UK_LCFS_DEVOLVED_BUS_FINANCE_RESOURCE,
    UK_NEED_ENERGY_FACTS_RESOURCE,
    UK_OFGEM_PRICE_CAP_RESOURCE,
)
#: Columns a declared rake levels; the support clip and the committed support
#: bounds leave them alone.
UK_LCFS_RAKED_COLUMNS = frozenset(
    {
        "electricity_consumption",
        "gas_consumption",
        "domestic_energy_consumption",
        "bus_fare_spending",
    }
)
UK_LCFS_ICE_SHARE_RULE = "one_minus_zero_emission_share"
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
#: Identity-keyed uniforms for the positive-regime bus fare draw of user
#: households the chain drew at zero (salted apart from the incidence draw).
UK_LCFS_BUS_FARE_POSITIVE_SALT = "lcfs_bus_fare_positive"
#: Columns whose level a declared rake sets; the support clip never touches them.
UK_LCFS_DEFAULT_SUPPORT_CLIP_EXEMPT = frozenset(
    {"electricity_consumption", "gas_consumption", "domestic_energy_consumption"}
)
UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS = (
    "is_adult",
    "is_child",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "hbai_household_net_income",
)
UK_LCFS_CONSUMPTION_PREDICTORS = (
    "is_adult",
    "is_child",
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
    bus_use_incidence: Mapping[str, Any] | None = None
    bus_fare_rake: Mapping[str, Any] | None = None
    energy_pricing: Mapping[str, Any] | None = None
    energy_rake: Mapping[str, Any] | None = None
    fuel_litres_audit: Mapping[str, Any] | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
            "stage": UK_LCFS_CONSUMPTION_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
        }
        if self.donor_uprating is not None:
            evidence["donor_uprating"] = dict(self.donor_uprating)
        if self.fuel_flag is not None:
            evidence["has_fuel_consumption"] = dict(self.fuel_flag)
        if self.bus_use_incidence is not None:
            evidence["bus_use_incidence"] = dict(self.bus_use_incidence)
        if self.bus_fare_rake is not None:
            evidence["bus_fare_rake"] = dict(self.bus_fare_rake)
        if self.energy_pricing is not None:
            evidence["energy_pricing"] = dict(self.energy_pricing)
        if self.energy_rake is not None:
            evidence["energy_rake"] = dict(self.energy_rake)
        if self.fuel_litres_audit is not None:
            evidence["fuel_litres_audit"] = dict(self.fuel_litres_audit)
        return evidence


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
        incidence = lcfs_bus_use_incidence(self.stage, frame)
        fare_rake_regions = bus_fare_rake_regions(self.stage)
        imputation = impute_lcfs_consumption(
            donor,
            recipient,
            seed=_operation_seed(self.stage, "fit_weighted_qrf_chain"),
            n_estimators=_qrf_n_estimators(self.stage),
            bus_users=None if incidence is None else incidence.household_user,
            bus_scope=None
            if incidence is None
            else np.isin(recipient["region"].astype(str).to_numpy(), fare_rake_regions),
            bus_positive_uniforms=None
            if incidence is None
            else stable_identity_uniforms(
                frame.table("household")["household_id"].to_numpy(),
                seed=_operation_seed(self.stage, "assign_bus_use_incidence"),
                salt=UK_LCFS_BUS_FARE_POSITIVE_SALT,
            ),
        )
        clip_result = support_clip_to_donor(
            imputation.draws, donor, exempt=support_clip_exempt(self.stage)
        )
        household_draws = clip_result.clipped
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
            )
        household_draws, bus_fare_rake_receipt = lcfs_bus_fare_rake(
            self.stage,
            household_draws,
            frame,
            recipient=recipient,
            users=None if incidence is None else incidence.household_user,
        )
        household_draws["domestic_energy_consumption"] = (
            household_draws["electricity_consumption"]
            + household_draws["gas_consumption"]
        )
        household_draws.loc[
            ~recipient["has_fuel_consumption"].astype(bool),
            ["petrol_spending", "diesel_spending"],
        ] = 0.0
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
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = imputation.fit_weight_records
        self.last_result = UKLCFSConsumptionResult(
            frame=result,
            support_clip=clip_result.receipt,
            donor_uprating=uprating_receipt,
            fuel_flag=fuel_flag_receipt,
            bus_use_incidence=None
            if incidence is None
            else {**incidence.receipt, **imputation.bus_fare_receipt},
            bus_fare_rake=bus_fare_rake_receipt,
            energy_pricing=None if energy is None else energy.receipt,
            energy_rake=energy_rake_receipt,
            fuel_litres_audit=litres_audit,
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
    bus_fare_receipt: Mapping[str, Any] = field(default_factory=dict)


def fuel_litres_audit(
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    stage: SourceStageSpec,
) -> dict[str, Any] | None:
    """Frame road-fuel litres against HMRC clearances times the OBR cars share.

    Diagnostic only (microcosm#890 C7): for each fuel column the declared
    uprating moves by the vendored litre proxy, the frame's design-weighted
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
        "gated": False,
    }


@dataclass(frozen=True)
class LCFSEnergyPricing:
    """The declared cap rates and NEED margins, with their receipts."""

    rates: EnergyCapRates
    margins: NeedMargins
    gas_connected: str
    receipt: dict[str, Any]


def lcfs_energy_pricing(stage: SourceStageSpec) -> LCFSEnergyPricing | None:
    """Resolve the declared ``price_energy_at_cap`` operation (none if undeclared)."""

    parameters = pricing_operation(stage)
    if parameters is None:
        return None
    columns = dict(parameters["columns"])
    if columns != {"electricity": "electricity_consumption", "gas": "gas_consumption"}:
        raise ValueError("price_energy_at_cap must price the two energy spend columns.")
    margins_resource = str(
        parameters.get("margins_resource") or UK_NEED_ENERGY_FACTS_RESOURCE
    )
    rates, rates_receipt = cap_rates(parameters)
    margins = need_margins_from_facts(margins_resource)
    return LCFSEnergyPricing(
        rates=rates,
        margins=margins,
        gas_connected=str(parameters.get("gas_connected", "positive_gas_spend")),
        receipt={**rates_receipt, "need_margins": margins.receipt},
    )


def energy_spend_to_kwh(
    table: pd.DataFrame, *, energy: LCFSEnergyPricing, region: np.ndarray
) -> pd.DataFrame:
    """Add ``electricity_kwh`` and ``gas_kwh`` at the declared regional cap rates."""

    result = table.copy()
    electricity = _numeric(result["electricity_consumption"]).to_numpy(dtype=float)
    gas = _numeric(result["gas_consumption"]).to_numpy(dtype=float)
    result[ELECTRICITY_KWH] = spend_to_kwh(
        electricity, frs_region=region, fuel="electricity", rates=energy.rates
    )
    result[GAS_KWH] = spend_to_kwh(
        gas, frs_region=region, fuel="gas", rates=energy.rates, connected=gas > 0
    )
    return result


def energy_kwh_to_spend(
    table: pd.DataFrame, *, energy: LCFSEnergyPricing, region: np.ndarray
) -> pd.DataFrame:
    """Price the kWh columns back to spend and drop them."""

    result = table.copy()
    gas_kwh = result[GAS_KWH].to_numpy(dtype=float)
    result["electricity_consumption"] = kwh_to_spend(
        result[ELECTRICITY_KWH].to_numpy(dtype=float),
        frs_region=region,
        fuel="electricity",
        rates=energy.rates,
    )
    result["gas_consumption"] = kwh_to_spend(
        gas_kwh,
        frs_region=region,
        fuel="gas",
        rates=energy.rates,
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Price the drawn energy spend to kWh, rake the four NEED margins, price back."""

    in_kwh = energy_spend_to_kwh(household_draws, energy=energy, region=region)
    raked, receipt = rake_energy_kwh(
        in_kwh,
        margins=energy.margins,
        frs_region=region,
        income=income,
        weights=weights,
        iterations=iterations,
        tenure=tenure,
        accommodation=accommodation,
        use_region_margin=True,
        gas_connected=in_kwh[GAS_KWH].to_numpy(dtype=float) > 0,
    )
    receipt["unit"] = "kwh"
    receipt["gas_connected_share"] = float(
        np.dot(
            raked[GAS_KWH].to_numpy(dtype=float) > 0, np.asarray(weights, dtype=float)
        )
        / max(float(np.sum(weights)), 1e-12)
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
    """The declared post-imputation energy IPF (the one with the region margin)."""

    operation = _energy_rake_operation(stage, margin="region")
    return 50 if operation is None else int(operation.parameters.get("iterations", 50))


def support_clip_exempt(stage: SourceStageSpec) -> set[str]:
    """Columns the declared ``support_clip`` exempts (raked columns keep their level)."""

    for operation in stage.operations:
        if operation.kind == "support_clip":
            declared = operation.parameters.get("exempt")
            if declared:
                return {str(column) for column in declared}
    return set(UK_LCFS_DEFAULT_SUPPORT_CLIP_EXEMPT)


def lcfs_bus_use_incidence(
    stage: SourceStageSpec, frame: Frame
) -> BusUseIncidenceResult | None:
    """Draw the declared local-bus use incidence on the recipient frame."""

    parameters = incidence_operation(stage)
    if parameters is None:
        return None
    if parameters.get("output") != "uses_local_bus":
        raise ValueError("assign_bus_use_incidence must output uses_local_bus.")
    shares, shares_receipt = nts_band_shares(parameters)
    result = assign_bus_use_incidence(
        frame.table("person"),
        frame.table("household"),
        household_weights=frame.weights_for("household").values,
        shares=shares,
        seed=int(parameters["seed"]),
    )
    return BusUseIncidenceResult(
        household_user=result.household_user,
        person_band=result.person_band,
        receipt={"nts": shares_receipt, **result.receipt},
    )


def bus_fare_rake_regions(stage: SourceStageSpec) -> tuple[str, ...]:
    """The FRS regions a declared bus-fare rake cell levels.

    The incidence override (positive-regime fill for users, zero for
    non-users) is imposed only there: a positive-regime amount is a
    diary-positive fortnight annualised, so without the rake to level it the
    override would overstate a region's fares several-fold. Where no
    publisher receipts exist (Wales) the chain's raw draw stands.
    """

    regions: list[str] = []
    for parameters in rake_operations(stage):
        if "bus_fare_spending" not in [str(c) for c in parameters["columns"]]:
            continue
        for cell in parameters["cells"]:
            regions.extend(str(region) for region in cell["regions"])
    return tuple(dict.fromkeys(regions))


def lcfs_bus_fare_rake(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    frame: Frame,
    *,
    recipient: pd.DataFrame,
    users: np.ndarray | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply every declared ``rake_to_vendored_facts`` operation on the stage."""

    operations = rake_operations(stage)
    if not operations:
        return household_draws, None
    household = frame.table("household")
    person = frame.table("person")
    persons = (
        person.groupby("person_household_id")
        .size()
        .reindex(household["household_id"])
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    receipts: list[dict[str, Any]] = []
    raked = household_draws
    for parameters in operations:
        cells = resolve_cells(parameters, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
        income_key = parameters.get("quintile_income")
        raked, receipt = rake_to_facts(
            raked,
            columns=[str(column) for column in parameters["columns"]],
            cells=cells,
            region=recipient["region"].astype(str).to_numpy(),
            weights=frame.weights_for("household").values
            if bool(parameters.get("weighted", True))
            else None,
            scope=str(parameters["scope"]),
            users=users,
            quintile_income=None
            if income_key is None
            else _numeric(recipient[str(income_key)]).to_numpy(dtype=float),
            household_persons=persons,
            iterations=int(parameters.get("iterations", 1)),
        )
        receipts.append(receipt)
    return raked, receipts[0] if len(receipts) == 1 else {"operations": receipts}


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
    """Design-weighted fuel-household shares on both sides of the imputation."""

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
    ``energy`` (the declared ``price_energy_at_cap``) converts the diary's
    energy spend to kWh at the FY2024-25 regional cap rates, rakes the income
    margin to the NEED means and prices back; without it the energy columns
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
    bus_users: np.ndarray | None = None,
    bus_positive_uniforms: np.ndarray | None = None,
    bus_scope: np.ndarray | None = None,
) -> UKLCFSConsumptionImputationResult:
    """Chain-draw the targets; optionally impose the bus-use incidence.

    With ``bus_users`` (one bool per recipient) the ``bus_fare_spending``
    draw is overridden after the chain step: non-user households take zero,
    user households drawn at zero take the step's positive-regime draw at
    their identity-keyed ``bus_positive_uniforms`` quantile, and user
    households drawn positive keep their draw. ``bus_scope`` limits the
    override to the households a declared fare rake levels afterwards (the
    regions with published receipts); outside it the raw draw stands. The
    chain itself keeps conditioning on the raw prior draw, so later targets
    are unchanged.
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
    bus_receipt: dict[str, Any] = {}
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
        if target == "bus_fare_spending" and bus_users is not None:
            draws[target], bus_receipt = _impose_bus_use_incidence(
                result,
                features.join(raw.drop(columns=[target])),
                raw_draw=np.asarray(result.raw_draw, dtype=float),
                users=np.asarray(bus_users, dtype=bool),
                uniforms=bus_positive_uniforms,
                scope=None if bus_scope is None else np.asarray(bus_scope, dtype=bool),
            )
        fit_records.append(
            FitWeightRecord(
                f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", result.weight_kind
            )
        )
        state = result.state
    return UKLCFSConsumptionImputationResult(draws, tuple(fit_records), bus_receipt)


def _impose_bus_use_incidence(
    step: Any,
    features: pd.DataFrame,
    *,
    raw_draw: np.ndarray,
    users: np.ndarray,
    uniforms: np.ndarray | None,
    scope: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if uniforms is None:
        raise ValueError("bus incidence needs identity-keyed positive-draw uniforms.")
    if len(users) != len(raw_draw) or len(uniforms) != len(raw_draw):
        raise ValueError("bus incidence inputs must align with the recipients.")
    in_scope = np.ones(len(raw_draw), dtype=bool) if scope is None else scope
    if len(in_scope) != len(raw_draw):
        raise ValueError("bus incidence scope must align with the recipients.")
    fitted = step.fitted
    if fitted is None:
        raise ValueError("the bus_fare_spending chain step exposes no fitted view.")
    positive = fitted.predict_positive_from_uniforms(
        features, quantiles={"bus_fare_spending": np.asarray(uniforms, dtype=float)}
    )["bus_fare_spending"].to_numpy(dtype=float)
    drawn_positive = raw_draw > 0
    filled = in_scope & users & ~drawn_positive
    imposed = np.where(users, np.where(drawn_positive, raw_draw, positive), 0.0)
    adjusted = np.where(in_scope, imposed, raw_draw)
    return adjusted, {
        "regime": str(getattr(step.regime, "name", step.regime)),
        "positive_draw_salt": UK_LCFS_BUS_FARE_POSITIVE_SALT,
        "chain_conditioned_on": "raw_draw",
        "households": int(len(raw_draw)),
        "households_in_scope": int(in_scope.sum()),
        "households_outside_scope_keep_raw_draw": int((~in_scope).sum()),
        "users": int(users.sum()),
        "users_in_scope": int((in_scope & users).sum()),
        "users_drawn_positive": int((in_scope & users & drawn_positive).sum()),
        "users_filled_from_positive_regime": int(filled.sum()),
        "non_users_zeroed": int((in_scope & ~users & drawn_positive).sum()),
        "positive_share_before": float(drawn_positive.mean()) if len(raw_draw) else 0.0,
        "positive_share_after": float((adjusted > 0).mean()) if len(raw_draw) else 0.0,
    }


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
    """Donor support per clipped column; raked columns carry no bounds."""

    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        if column in UK_LCFS_RAKED_COLUMNS:
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
