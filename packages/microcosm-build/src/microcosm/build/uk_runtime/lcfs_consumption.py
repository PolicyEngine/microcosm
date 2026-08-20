"""UK LCFS consumption imputation stage."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.raking import MarginSpec, iterative_proportional_fit
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.was_wealth import (
    clean_was_household_table,
    encode_qrf_predictor_pair,
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
UK_LCFS_HAS_FUEL_PREDICTORS = (
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "region",
)
# LCFS-native names for the three bridge predictors the WAS donor names
# differently (incumbent consumption.py:556-574).
LCFS_TO_WAS_HAS_FUEL_RENAMES = {
    "hbai_household_net_income": "household_net_income",
    "is_adult": "num_adults",
    "is_child": "num_children",
}
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
    "has_fuel_consumption",
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
UK_LCFS_HAS_FUEL_FIT_NAME = "uk_was_2018_20_has_fuel"


@dataclass
class UKLCFSConsumptionStageTransform:
    """Whole-stage callable for LCFS-trained consumption imputation."""

    stage: SourceStageSpec
    engine: object
    lcfs_hh_tab_path: str | Path | None = None
    lcfs_person_tab_path: str | Path | None = None
    was_tab_path: str | Path | None = None
    lcfs_household: pd.DataFrame | None = None
    lcfs_person: pd.DataFrame | None = None
    was_donor: pd.DataFrame | None = None
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        anchors = load_lcfs_consumption_anchors()
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
        was_raw = (
            self.was_donor
            if self.was_donor is not None
            else read_pinned_tab(
                _require_path(self.was_tab_path),
                _artifact(self.stage, "was_bridge_donor"),
            )
        )
        was = clean_was_household_table(was_raw)
        donor = clean_lcfs_consumption_table(lcfs_person, lcfs_household)
        donor, bridge_record = bridge_has_fuel_to_lcfs(
            donor,
            was,
            seed=_operation_seed(self.stage, "bridge_donor_column_via_qrf"),
            nts_ice_share=float(anchors["nts_ice_share"]["value"]),
        )
        recipient = recipient_predictors(frame, self.engine)
        recipient["has_fuel_consumption"] = assign_recipient_has_fuel(
            frame,
            rate=float(anchors["nts_ice_share"]["value"]),
            seed=_operation_seed(self.stage, "assign_binary_from_rate"),
        )
        imputation = impute_lcfs_consumption(
            donor,
            recipient,
            seed=_operation_seed(self.stage, "fit_weighted_qrf_chain"),
            n_estimators=_qrf_n_estimators(self.stage),
        )
        household_draws = support_clip_to_donor(
            imputation.draws,
            donor,
            exempt={
                "electricity_consumption",
                "gas_consumption",
                "domestic_energy_consumption",
            },
        )
        household_draws = rake_energy_to_need(
            household_draws.join(recipient[["household_gross_income"]]),
            weights=frame.weights_for("household").values,
            tenure=recipient["tenure_type"].astype(str).to_numpy(),
            accommodation=recipient["accommodation_type"].astype(str).to_numpy(),
            region=recipient["region"].astype(str).to_numpy(),
        )
        household_draws["domestic_energy_consumption"] = (
            household_draws["electricity_consumption"]
            + household_draws["gas_consumption"]
        )
        household_draws.loc[
            ~recipient["has_fuel_consumption"].astype(bool),
            ["petrol_spending", "diesel_spending"],
        ] = 0.0
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
        self.last_fit_weight_records = (bridge_record, *imputation.fit_weight_records)
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS


@dataclass(frozen=True)
class UKLCFSConsumptionImputationResult:
    draws: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]


def clean_lcfs_consumption_table(
    lcfs_person: pd.DataFrame, lcfs_household: pd.DataFrame
) -> pd.DataFrame:
    """Return the LCFS donor table with annualized consumption variables."""

    person = _lowercase(lcfs_person).rename(columns=PERSON_LCFS_RENAMES)
    household = _lowercase(lcfs_household).rename(columns=HOUSEHOLD_LCFS_RENAMES)
    _require_columns(household, ("case", *HOUSEHOLD_LCFS_RENAMES.values()))
    _require_columns(person, ("case", *PERSON_LCFS_RENAMES.values()))
    household["region"] = _numeric(household["region"]).map(LCFS_REGIONS)
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
    household = rake_energy_to_need(household, weights=None, iterations=1)
    household["domestic_energy_consumption"] = (
        household["electricity_consumption"] + household["gas_consumption"]
    )
    return household[
        [
            *UK_LCFS_CONSUMPTION_PREDICTORS[:-1],
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


def bridge_has_fuel_to_lcfs(
    lcfs: pd.DataFrame,
    was: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = 100,
    nts_ice_share: float | None = None,
) -> tuple[pd.DataFrame, FitWeightRecord]:
    """Fit a WAS has-fuel bridge and predict a clipped rate onto LCFS."""

    from microcosm.fit import RegimeGatedQRF

    if nts_ice_share is None:
        nts_ice_share = float(
            load_lcfs_consumption_anchors()["nts_ice_share"]["value"]
        )
    donor = was.copy()
    donor["has_fuel_consumption"] = (
        (_numeric(donor["num_vehicles"]) > 0)
        & (
            stable_identity_uniforms(
                donor.index.to_numpy(), seed=seed, salt="was_has_fuel"
            )
                < nts_ice_share
        )
    ).astype(float)
    # The LCFS frame carries its own names for three of the WAS bridge
    # predictors (incumbent consumption.py:556-574 renames before predicting).
    recipient = lcfs.rename(columns=LCFS_TO_WAS_HAS_FUEL_RENAMES)
    donor_encoded, recipient_encoded, predictors = encode_qrf_predictor_pair(
        donor[[*UK_LCFS_HAS_FUEL_PREDICTORS, "has_fuel_consumption", "weight"]],
        recipient[list(UK_LCFS_HAS_FUEL_PREDICTORS)],
        predictors=UK_LCFS_HAS_FUEL_PREDICTORS,
    )
    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    result = model.fit(
        donor_encoded,
        list(predictors),
        ["has_fuel_consumption"],
        weights="weight",
    ).predict(recipient_encoded)
    out = lcfs.copy()
    out["has_fuel_consumption"] = np.clip(
        np.asarray(result["has_fuel_consumption"], dtype=float), 0.0, 1.0
    )
    return out, FitWeightRecord(UK_LCFS_HAS_FUEL_FIT_NAME, "explicit")


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
    for predictor in ("region", "tenure_type", "accommodation_type", "num_vehicles"):
        if predictor in household:
            result[predictor] = household[predictor].map(_enum_name).to_numpy()
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
    fit_records: list[FitWeightRecord] = []
    for target in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        result = model.fit_draw_next(
            donor_encoded,
            recipient_encoded.loc[:, list(predictors)],
            raw,
            state=state,
            weights="household_weight",
        )
        raw[target] = result.raw_draw
        fit_records.append(
            FitWeightRecord(
                f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", result.weight_kind
            )
        )
        state = result.state
    return UKLCFSConsumptionImputationResult(raw, tuple(fit_records))


def support_clip_to_donor(
    draws: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    exempt: set[str] | None = None,
) -> pd.DataFrame:
    clipped = draws.copy()
    exempt = exempt or set()
    for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        if column in exempt or column not in clipped or column not in donor:
            continue
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        clipped[column] = clipped[column].clip(
            lower=float(finite.min()), upper=float(finite.max())
        )
    return clipped


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        if column in {
            "electricity_consumption",
            "gas_consumption",
            "domestic_energy_consumption",
        }:
            continue
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


def rake_energy_to_need(
    household: pd.DataFrame,
    *,
    weights: Sequence[float] | None,
    iterations: int = 50,
    tenure: Sequence[str] | None = None,
    accommodation: Sequence[str] | None = None,
    region: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rake electricity/gas to the NEED margins.

    The donor-side single-pass call rakes the income margin only (the
    incumbent's training-side calibration). The post-imputation call passes
    all four groupers and sweeps income -> tenure -> accommodation -> region
    per iteration, the incumbent's order. Categories absent from the NEED
    maps (CONVERTED_HOUSE/OTHER/UNKNOWN accommodation; Scotland and Northern
    Ireland regions) are deliberately untouched by that margin.
    """

    frame = household.copy()
    frame["_need_income_band"] = _income_band(frame["household_gross_income"])
    margins = [MarginSpec("_need_income_band", _need_income_targets())]
    scratch = ["_need_income_band"]
    for name, values in (
        ("tenure", tenure),
        ("accommodation", accommodation),
        ("region", region),
    ):
        if values is None:
            continue
        column = f"_need_{name}"
        frame[column] = np.asarray(values).astype(str)
        targets, _ = _need_categorical_targets(name)
        margins.append(MarginSpec(column, targets))
        scratch.append(column)
    weight_column = None
    if weights is not None:
        frame["_weight"] = np.asarray(weights, dtype=float)
        weight_column = "_weight"
        scratch.append("_weight")
    raked = iterative_proportional_fit(
        frame,
        columns=("electricity_consumption", "gas_consumption"),
        margins=tuple(margins),
        iterations=iterations,
        weight_column=weight_column,
    )
    return raked.drop(columns=[c for c in scratch if c in raked])


def load_lcfs_consumption_anchors() -> dict:
    from importlib.resources import files

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("lcfs_consumption_anchors.json")
        .read_text(encoding="utf-8")
    )


def _load_need_energy_targets() -> dict:
    from importlib.resources import files

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("need_energy_targets.json")
        .read_text(encoding="utf-8")
    )


def _need_income_bands() -> tuple[tuple[float, float, str, float, float], ...]:
    bands = []
    for band in _load_need_energy_targets()["income_bands"]:
        upper = np.inf if band["upper"] is None else float(band["upper"])
        bands.append(
            (
                float(band["lower"]),
                upper,
                str(band["label"]),
                float(band["gas_kwh"]),
                float(band["electricity_kwh"]),
            )
        )
    return tuple(bands)


def _need_income_targets() -> dict:
    rates = _load_need_energy_targets()["source"]["ofgem_q2_2026"]
    gas_rate = float(rates["gas_gbp_per_kwh"])
    electricity_rate = float(rates["electricity_gbp_per_kwh"])
    return {
        name: {
            "gas_consumption": gas * gas_rate,
            "electricity_consumption": electricity * electricity_rate,
        }
        for _, _, name, gas, electricity in _need_income_bands()
    }


def _need_categorical_targets(margin: str) -> tuple[dict, dict]:
    """(category -> column -> spend target, frs-value -> need-key map).

    Built from the committed NEED resource so the raking and the
    aggregate_admin anchors share one source of values.
    """

    need = _load_need_energy_targets()
    rates = need["source"]["ofgem_q2_2026"]
    gas_rate = float(rates["gas_gbp_per_kwh"])
    electricity_rate = float(rates["electricity_gbp_per_kwh"])
    block = need[margin]
    if margin == "region":
        mapping = {name: name for name in block["gas_kwh"]}
    else:
        mapping = dict(block["map"])
    targets = {
        frs_value: {
            "gas_consumption": block["gas_kwh"][need_key] * gas_rate,
            "electricity_consumption": (
                block["electricity_kwh"][need_key] * electricity_rate
            ),
        }
        for frs_value, need_key in mapping.items()
    }
    return targets, mapping


def _income_band(values: pd.Series) -> pd.Series:
    income = _numeric(values)
    result = pd.Series(index=income.index, dtype=object)
    for lo, hi, name, _, _ in _need_income_bands():
        result[(income >= lo) & (income < hi)] = name
    return result


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
