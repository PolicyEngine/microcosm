"""UK simulated-measure inputs for the national calibration seam."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from datetime import date
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.atomic_area_support import (
    without_uk_native_alias_columns,
)
from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_EMPLOYEE_MAIN_SOURCE,
    CENTAX_EMPLOYEE_SOURCE_INPUTS,
    CENTAX_PARENT_ENGINE_INPUTS,
    CENTAX_PARENT_MEASURES,
    CENTAX_PARENT_OF_ELIGIBLE_3_4,
    employment_main_income_source,
    parent_of_eligible_3_4_year_old,
)
from microcosm.build.uk_runtime.cgt_calibration import UK_CGT_DATED_MEASURE
from microcosm.build.uk_runtime.dfe_spring_term import (
    DFE_SPRING_TERM_ENGINE_INPUTS,
    DFE_SPRING_TERM_MEASURES,
    dfe_spring_term_status,
)
from microcosm.build.uk_runtime.domestic_energy_targets import (
    household_energy_kwh,
    household_energy_prices,
)
from microcosm.build.uk_runtime.energy_pricing import ELECTRICITY_KWH, GAS_KWH
from microcosm.build.uk_runtime.frs_annual_income import (
    CONTRIBUTORY_BENEFITS_ANNUAL_SHARE,
    EMPLOYMENT_INCOME_ANNUAL_SHARE,
    EMPLOYMENT_INCOME_OTHER_SPELLS,
    FRS_ANNUAL_INCOME_OUTPUT_COLUMNS,
    PRIVATE_PENSION_INCOME_ANNUAL_SHARE,
    SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE,
    SELF_EMPLOYMENT_INCOME_OTHER_SPELLS,
)
from microcosm.build.uk_runtime.high_value_homes import (
    high_value_home_mask,
    uk_high_value_home_thresholds,
)
from microcosm.build.uk_runtime.hmrc_income import (
    LOSS_RELIEF_COLUMN,
    self_employment_income_after_loss_relief,
)
from microcosm.build.uk_runtime.income_tax_rate_classes import (
    UK_RATE_CLASS_ALLOCATION,
    UK_RATE_CLASS_ENGINE_FLOAT32_STEPS,
    UK_RATE_CLASS_ENGINE_TOLERANCE_GBP,
    UK_RATE_CLASS_ENGINE_VARIABLE,
    UK_RATE_CLASSES,
    rate_class_engine_tolerance,
    relief_by_rate_class,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_BUILD_IDENTIFIER_COLUMNS,
    UK_RELEASE_EXPORT_DROPPED_COLUMNS,
    UK_TIME_PERIOD_METADATA_KEY,
    load_uk_national_frame,
    uk_release_identifier_tables,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.uc_relationships import (
    frs_uc_claimant_mask,
    uc_family_child_member,
)
from microcosm.build.uk_runtime.uc_target_measurements import (
    UC_PAID_TARGET_VARIABLES,
    UC_TARGET_VARIABLES,
    compute_uc_target_measure,
    uc_administrative_family_type,
    uc_child_entitlement,
    uc_paid_comparison_contract,
    uc_tcl_comparison_contract,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    exclusion_evaluation_date,
)
from microcosm.calibrate import TargetRegistry
from microcosm.frame import Frame, MassChangeRecord, Weights
from microcosm.frame.adapters.policyengine_uk import validate_uc_claimant_input

_ENTITY_LINK = {"benunit": "person_benunit_id", "household": "person_household_id"}
_ENTITY_ID = {
    "person": "person_id",
    "benunit": "benunit_id",
    "household": "household_id",
}
_UC_CALIBRATION_VARIABLES = frozenset(
    {"uc_calibration_family_type", "uc_calibration_child_count"}
)

# Reserved, transient measurement names: using the model's input name here
# would let a persisted base-period column bypass requested-year calculation.
# Both the taxpayer mask and amount must use the same engine-year net gains.
_CGT_CALIBRATION_VARIABLES = {
    "cgt_calibration_gains": "capital_gains",
    "cgt_calibration_tax": "capital_gains_tax",
    # The BADR/Investors' Relief qualifying gain the asset-type stage writes
    # (microcosm#1014); banded and filtered on for the Table 4.1 rows.
    "cgt_calibration_badr_gains": "capital_gains_badr",
    # Derived, not a policyengine-uk variable: see _cgt_taxable_income.
    "cgt_calibration_taxable_income": "cgt_taxable_income",
}

#: The taxable income policyengine-uk's ``capital_gains_tax`` sets the CGT
#: rate bands on, which the engine computes inline and exposes as no
#: variable; HMRC's Table 3 bands individuals on the same concept
#: (microcosm#1014). Derived from these engine variables at the measure year.
_CGT_TAXABLE_INCOME = "cgt_taxable_income"
_CGT_TAXABLE_INCOME_COMPONENTS = (
    "net_income",
    "allowances",
    "personal_pension_contributions",
    "pension_contributions_relief",
)
_CGT_TAXABLE_INCOME_DEFINITION = (
    "max(0, net_income - max(0, allowances - "
    "min(personal_pension_contributions, pension_contributions_relief))), as "
    "policyengine-uk's capital_gains_tax computes the taxable income it stacks "
    "gains on; from 2.128.0 Gift Aid extends the basic rate band instead of "
    "reducing allowances (pe-uk#2271)"
)


def cgt_taxable_income_from_components(parts: Mapping[str, Any]) -> np.ndarray:
    """The engine CGT formula's taxable income from its component arrays.

    policyengine-uk 2.128.0 and later stack gains on net income less the
    allowances, with relief-at-source pension contributions taken off the
    allowances; Gift Aid extends the basic rate band instead (pe-uk#2271), so
    it moves where the band ends, not the income the band is measured on.
    """

    values = {
        name: np.asarray(parts[name], dtype=float)
        for name in _CGT_TAXABLE_INCOME_COMPONENTS
    }
    band_extension = np.minimum(
        values["personal_pension_contributions"],
        values["pension_contributions_relief"],
    )
    allowances = np.maximum(0.0, values["allowances"] - band_extension)
    return np.maximum(0.0, values["net_income"] - allowances)


#: DfT BUS01 boardings from the NTS trips the frame carries (microcosm#1207):
#: each household's trips on the series times the boardings per trip the
#: lcfs stage priced that series at, by the household's region (Wales is
#: unpriced and no row binds it). Reserved names: a persisted column of
#: either would bypass the stage's factor.
_BUS_BOARDING_MEASURES = {
    "bus_in_london_boardings": "bus_in_london_trips",
    "other_local_bus_boardings": "other_local_bus_trips",
}


def household_bus_boardings(frame: Any, variable: str) -> np.ndarray:
    """A household's boardings on one bus series, in the frame's household order."""

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.bus_fare_pricing import bus_fare_prices
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_VENDORED_RESOURCES,
        bus_pricing_operation,
    )

    trips_column = _BUS_BOARDING_MEASURES[variable]
    declared = bus_pricing_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    if declared is None:
        raise KeyError("the lcfs stage declares no bus pricing")
    prices = bus_fare_prices(declared, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    person = frame.table("person")
    household = frame.table("household")
    trips = person[trips_column].to_numpy(dtype=float)
    if not np.isfinite(trips).all() or (trips < 0).any():
        raise ValueError(f"{trips_column} must be finite and non-negative.")
    if variable == "bus_in_london_boardings":
        # London's buses carry every resident's bus-in-London trips, wherever
        # they live: the series is priced once, at London's factor.
        per_trip = np.full(len(person), prices.london.boardings_per_trip)
    else:
        region_by_household = pd.Series(
            household["region"].astype(str).to_numpy(),
            index=household["household_id"].to_numpy(),
        )
        region = region_by_household.loc[
            person["person_household_id"].to_numpy()
        ].to_numpy()
        per_trip = np.asarray(
            [
                prices.other_by_region[name].boardings_per_trip
                if name in prices.other_by_region
                else 0.0
                for name in region
            ]
        )
    boardings = (
        pd.Series(trips * per_trip)
        .groupby(person["person_household_id"].to_numpy())
        .sum()
    )
    return boardings.reindex(
        household["household_id"].to_numpy(), fill_value=0.0
    ).to_numpy(dtype=float)


#: Owner-occupied homes over each high-value threshold, one indicator per
#: household: the measure of the region rows that control the family's area
#: cells on the local line (microcosm#1301). The indicator is the stored
#: main_residence_value and tenure_type through ``high_value_home_mask`` at the
#: threshold's frame-basis value, as the local metrics and the release gates
#: read them, never an engine-uprated value. Reserved names: a persisted column
#: of any would bypass the mask.
_HIGH_VALUE_HOME_MEASURES = {
    "owner_occupied_home_over_gbp_1m": "over_gbp_1m",
    "owner_occupied_home_over_gbp_1_5m": "over_gbp_1_5m",
    "owner_occupied_home_over_gbp_2m": "over_gbp_2m",
}
_HIGH_VALUE_HOME_COLUMNS = ("main_residence_value", "tenure_type")


def household_high_value_home(frame: Any, variable: str) -> np.ndarray:
    """One for an owner-occupied household whose stored home value is over the threshold."""

    threshold_id = _HIGH_VALUE_HOME_MEASURES[variable]
    threshold = {
        candidate.threshold_id: candidate
        for candidate in uk_high_value_home_thresholds()
    }[threshold_id]
    household = frame.table("household")
    missing = [column for column in _HIGH_VALUE_HOME_COLUMNS if column not in household]
    if missing:
        raise ValueError(f"{variable} needs the stored household columns {missing}.")
    return high_value_home_mask(
        household["main_residence_value"].to_numpy(dtype=float),
        household["tenure_type"].to_numpy(),
        threshold_basis_gbp=threshold.basis_gbp,
    ).astype(float)


#: HMRC's statistics count the year's profit after losses brought forward,
#: which policyengine-uk keeps apart in loss_relief: the self-employment rows
#: read that profit, and every row on HMRC's total income (its bands, its
#: amounts, its regional filters) reads total income with that profit
#: (microcosm#1095). The engine's total_income adds the profit before the
#: relief, so the total-income measure swaps the one for the other.
UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF = "self_employment_income_after_loss_relief"
UK_TOTAL_INCOME_AFTER_LOSS_RELIEF = "total_income_after_loss_relief"
_LOSS_RELIEF_MEASURES = (
    UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF,
    UK_TOTAL_INCOME_AFTER_LOSS_RELIEF,
)
_LOSS_RELIEF_MEASURE_COMPONENTS = (
    "self_employment_income",
    LOSS_RELIEF_COLUMN,
    "total_income",
)


#: DESNZ's sub-national domestic energy (microcosm#1207): each household's
#: electricity and gas kWh, and whether it is gas-connected. The lcfs stage
#: stores each household's calendar-2024 spend; read at the dataset's period
#: the engine does not uprate it, the stage's own prices turn it back into
#: the kWh it raked, and a household that pays for gas is gas-connected.
_GAS_CONNECTED = "gas_connected"
UK_HOUSEHOLD_ENERGY_MEASURES: Mapping[str, str] = {
    "domestic_electricity_kwh": ELECTRICITY_KWH,
    "domestic_gas_kwh": GAS_KWH,
    "domestic_gas_connected": _GAS_CONNECTED,
}


#: HMRC's income statistics count the income each taxpayer received over a
#: tax year, banded by what they received; they cover individuals liable to UK
#: income tax. The frame's incomes are current, annualised ones, and
#: ``frs_annual_income_spells`` stores beside them what each adult received
#: over the 12 months before the interview. Each annual measure is its current
#: measure computed on a second simulation whose inputs are each person's year,
#: so the engine's own rules give the tax on it. That year is a proxy for the
#: tax year: the 12 months before the interview, applied to the engine's
#: amounts at the calibration year (the stage module says how). Poverty,
#: benefits and the engine's published tax keep reading current income.
#: CenTax's tabulations of parents' adjusted net income come from the same
#: HMRC data, so they band on the year's ANI too.
UK_ANNUAL_INCOME_MEASURES: Mapping[str, str] = {
    "employment_income_annual": "employment_income",
    "self_employment_income_after_loss_relief_annual": (
        UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF
    ),
    "private_pension_income_annual": "private_pension_income",
    "state_pension_annual": "state_pension",
    "total_income_after_loss_relief_annual": UK_TOTAL_INCOME_AFTER_LOSS_RELIEF,
    "income_tax_annual": "income_tax",
    "tax_band_annual": "tax_band",
    "adjusted_net_income_annual": "adjusted_net_income",
}
#: The inputs the annual simulation replaces at the measurement year: the
#: engine's current amount times its annual share, plus its other spells,
#: carried from the dataset's year by the engine's own factor for the input.
_ANNUAL_INCOME_INPUTS: tuple[tuple[str, str, str | None], ...] = (
    (
        "employment_income_before_lsr",
        EMPLOYMENT_INCOME_ANNUAL_SHARE,
        EMPLOYMENT_INCOME_OTHER_SPELLS,
    ),
    (
        "self_employment_income",
        SELF_EMPLOYMENT_INCOME_ANNUAL_SHARE,
        SELF_EMPLOYMENT_INCOME_OTHER_SPELLS,
    ),
    ("private_pension_income", PRIVATE_PENSION_INCOME_ANNUAL_SHARE, None),
    ("jsa_contrib_reported", CONTRIBUTORY_BENEFITS_ANNUAL_SHARE, None),
    ("esa_contrib_reported", CONTRIBUTORY_BENEFITS_ANNUAL_SHARE, None),
    ("incapacity_benefit_reported", CONTRIBUTORY_BENEFITS_ANNUAL_SHARE, None),
)
#: Taxable benefits whose entitlement reads earnings, held at their baseline
#: values: a carer was paid in the year at the earnings of the time, which the
#: current concept records, not at the year's total.
_ANNUAL_INCOME_HELD_VARIABLES = (
    "carers_allowance",
    "carer_support_payment",
    "scottish_carer_supplement",
)
#: ``months_since_state_pension_age`` counts to the middle of the fiscal year
#: (6 October), so someone attaining State Pension age within the year was
#: paid for that many months plus six of it.
_FISCAL_HALF_YEAR_MONTHS = 6.0


def state_pension_annual_share(months_since_state_pension_age: Any) -> np.ndarray:
    """The share of the fiscal year a State Pension starting in it was paid.

    Only someone attaining State Pension age within the year, less than six
    months either side of its middle, is paid for part of it. Anyone else
    keeps the whole year, an amount reported below the age included: the
    current concept carries it, and the descriptors say nothing about it.
    """

    months = np.asarray(months_since_state_pension_age, dtype=float)
    in_year = np.isfinite(months) & (np.abs(months) < _FISCAL_HALF_YEAR_MONTHS)
    share = np.ones(months.shape)
    share[in_year] = (months[in_year] + _FISCAL_HALF_YEAR_MONTHS) / (
        2.0 * _FISCAL_HALF_YEAR_MONTHS
    )
    return share


#: Every binding key that names a measured variable: a dated CGT measure in
#: any of them must carry its own measurement period (microcosm#1014 bands
#: and filters on dated measures, not only gates and values).
_BINDING_VARIABLE_KEYS = ("gated_variable", "value_variable", "groupby_variable")


def _binding_variable_names(binding: Mapping[str, Any]) -> list[str]:
    names = [str(binding.get(key, "")) for key in _BINDING_VARIABLE_KEYS]
    for key in ("filters", "household_conditions"):
        for predicate in binding.get(key, ()) or ():
            if isinstance(predicate, Mapping):
                names.append(str(predicate.get("variable", "")))
    return [name for name in names if name]


def _cgt_model_measure(variable: str, default_year: int) -> tuple[str, int] | None:
    if variable in _CGT_CALIBRATION_VARIABLES:
        return _CGT_CALIBRATION_VARIABLES[variable], default_year
    match = UK_CGT_DATED_MEASURE.fullmatch(variable)
    if match:
        return _CGT_CALIBRATION_VARIABLES[f"cgt_calibration_{match[2]}"], int(match[1])
    return None


def _uc_calibration_composition(
    frame: Any, simulation: Any, year: int
) -> tuple[np.ndarray, np.ndarray]:
    """Measure FRS families without promoting an older child to a partner.

    FRS ``is_parent`` identifies adult-file members of benefit units with
    children, including unmarried partners; ``is_married`` alone would miss
    cohabiting couples. In units without recorded children all members are
    the single claimant or couple. These retained roles survive SPI cloning.

    DWP's post-April-2019 child-presence definition includes reported under-20
    children beyond those eligible for a UC child element. Include those FRS
    child members as well as UC qualifying children, never the claimants:
    https://stat-xplore.dwp.gov.uk/webapi/metadata/UC_Households/Family%20Type.html
    This is a calibration composition proxy, not a change to UC entitlement.
    """
    person = frame.table("person")
    benunit = frame.table("benunit")
    member_ids = person["person_benunit_id"].to_numpy()
    ids = benunit["benunit_id"].to_numpy()
    claimant = frs_uc_claimant_mask(person, benunit)
    claimant_count = (
        pd.Series(claimant).groupby(member_ids).sum().reindex(ids).to_numpy()
    )
    qualifying = _values(
        simulation.calculate(
            "is_child_or_qualifying_young_person_for_universal_credit", year
        )
    ).astype(bool)
    if len(qualifying) != len(person):
        raise ValueError("UC child status must align with the person table.")
    children = uc_family_child_member(
        claimant, qualifying, person["age"].to_numpy(dtype=float)
    )
    child_count = pd.Series(children).groupby(member_ids).sum().reindex(ids).to_numpy()
    couple = claimant_count == 2
    has_children = child_count > 0
    family_type = np.select(
        [couple & has_children, couple, has_children],
        ["COUPLE_WITH_CHILDREN", "COUPLE_NO_CHILDREN", "LONE_PARENT"],
        default="SINGLE",
    )
    return family_type, child_count.astype(float)


def _measured_at_frame_period(frame: Any, year: int | str) -> bool:
    """Whether ``year`` is the period the frame's stored columns hold.

    A frame with no period label is treated as holding the requested year,
    which keeps the frame route of hermetic frames that carry no metadata.
    """

    metadata = getattr(frame, "metadata", None) or {}
    period = metadata.get(UK_TIME_PERIOD_METADATA_KEY)
    if not isinstance(period, str) or not period.strip():
        return True
    return period == str(year)


def compute_uk_measure_input(
    frame: Any, simulation: Any, entity: str, variable: str, year: int
) -> tuple[np.ndarray, str]:
    """Compute one policyengine-uk variable at the requested entity grain."""

    cgt_measure = _cgt_model_measure(variable, year)
    if cgt_measure is not None:
        if entity != "person":
            raise KeyError(f"CGT calibration measures are person-only: {entity}")
        model_variable, measure_year = cgt_measure
        values, route = compute_uk_measure_input(
            frame, simulation, entity, model_variable, measure_year
        )
        return values, f"engine_period:{measure_year}:{model_variable}:{route}"

    if variable == _CGT_TAXABLE_INCOME:
        if entity != "person":
            raise KeyError(f"CGT taxable income is person-only: {entity}")
        parts = {
            name: compute_uk_measure_input(frame, simulation, "person", name, year)[0]
            for name in _CGT_TAXABLE_INCOME_COMPONENTS
        }
        return (
            cgt_taxable_income_from_components(parts),
            "engine_components:capital_gains_tax_taxable_income",
        )

    if variable in _BUS_BOARDING_MEASURES:
        if entity != "household":
            raise KeyError(f"bus boardings are household-only: {entity}")
        return (
            household_bus_boardings(frame, variable),
            f"stage_boardings_per_trip:{_BUS_BOARDING_MEASURES[variable]}",
        )

    if variable in _HIGH_VALUE_HOME_MEASURES:
        if entity != "household":
            raise KeyError(f"high-value home measures are household-only: {entity}")
        return (
            household_high_value_home(frame, variable),
            f"frame_columns:high_value_home_mask:{_HIGH_VALUE_HOME_MEASURES[variable]}",
        )

    if variable in DFE_SPRING_TERM_MEASURES:
        if entity != "person":
            raise KeyError(f"DfE spring-term measures are person-only: {entity}")
        status = dfe_spring_term_status(
            lambda name: compute_uk_measure_input(
                frame, simulation, "person", name, year
            )[0],
            year,
        )
        return status[variable], f"dfe_spring_term:{variable}"

    if variable == CENTAX_PARENT_OF_ELIGIBLE_3_4:
        if entity != "person":
            raise KeyError(f"{variable} is person-only: {entity}")
        inputs = {
            name: compute_uk_measure_input(frame, simulation, "person", name, year)[0]
            for name in CENTAX_PARENT_ENGINE_INPUTS
        }
        return (
            parent_of_eligible_3_4_year_old(
                birth_year=inputs["birth_year"],
                country=inputs["country"],
                term_share_before_school=inputs["early_years_term_share_before_school"],
                dependent_child=inputs["is_hbai_dependent_child"],
                benunit_id=frame.table("person")[_ENTITY_LINK["benunit"]].to_numpy(),
                year=year,
            ),
            f"engine_terms_and_benunit_roles:{variable}",
        )
    if variable == CENTAX_EMPLOYEE_MAIN_SOURCE:
        raise KeyError(
            f"{variable} compares the year's income sources; resolve it through "
            "UKMeasureResolver, which computes the annual measures."
        )

    if variable in _LOSS_RELIEF_MEASURES:
        if entity != "person":
            raise KeyError(f"{variable} is person-only: {entity}")
        person = frame.table("person")
        if (
            variable == UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF
            and "self_employment_income" in person
            and _measured_at_frame_period(frame, year)
        ):
            # At the frame's own period the stored columns are the values:
            # the profit after the relief is read from them (microcosm#1095).
            # At any other period the engine computes both components there,
            # as it does every other stored value (microcosm#1324).
            stored_relief = (
                person[LOSS_RELIEF_COLUMN].to_numpy(dtype=float)
                if LOSS_RELIEF_COLUMN in person
                else np.zeros(len(person))
            )
            return (
                self_employment_income_after_loss_relief(
                    person["self_employment_income"].to_numpy(dtype=float),
                    stored_relief,
                ),
                "frame_columns:self_employment_income_less_loss_relief",
            )
        return _loss_relief_measure_from_engine(frame, simulation, variable, year)

    if variable in UC_TARGET_VARIABLES:
        if entity != "benunit":
            raise KeyError(f"UC statistical measurements are benunit-only: {entity}")
        return compute_uc_target_measure(
            frame, simulation, variable, year
        ), "uc_claim_statistical_proxy"
    if variable == "uc_calibration_child_entitlement":
        if entity != "benunit":
            raise KeyError(f"UC child entitlement is benunit-only: {entity}")
        return uc_child_entitlement(
            frame, simulation, year
        ), "uc_child_element_entitlement_proxy"
    if variable in _UC_CALIBRATION_VARIABLES | UC_PAID_TARGET_VARIABLES:
        if entity != "benunit":
            raise KeyError(f"UC calibration composition is benunit-only: {entity}")
        family_type, child_count = _uc_calibration_composition(frame, simulation, year)
        if variable == "uc_calibration_administrative_family_type":
            return uc_administrative_family_type(
                frame, simulation, year, child_count
            ), "uc_allowance_and_reported_child_proxy"
        values = (
            family_type if variable == "uc_calibration_family_type" else child_count
        )
        return values, "frs_relationships_and_uc_child_status"

    definition = simulation.tax_benefit_system.variables.get(variable)
    if definition is None:
        raise KeyError(f"policyengine-uk has no variable {variable!r}")
    native = definition.entity.key
    table = frame.table(entity)
    raw = simulation.calculate(variable, year)

    if native == entity:
        values = _values(raw)
        route = "native"
    else:
        native_values = _values(raw)
        if native_values.dtype.kind in {"O", "U", "S", "b"}:
            if entity == "person" and native in ("benunit", "household"):
                native_table = frame.table(native)
                lookup = pd.Series(
                    native_values, index=native_table[_ENTITY_ID[native]].to_numpy()
                )
                keys = frame.table("person")[_ENTITY_LINK[native]].to_numpy()
                values = lookup.loc[keys].to_numpy()
                route = f"categorical_broadcast_{native}_to_person"
            elif native == "person" and native_values.dtype.kind == "b":
                person = frame.table("person")
                collapsed = (
                    pd.Series(native_values.astype(bool))
                    .groupby(person[_ENTITY_LINK[entity]].to_numpy())
                    .max()
                )
                keys = table[_ENTITY_ID[entity]].to_numpy()
                values = collapsed.loc[keys].to_numpy().astype(float)
                route = f"bool_any_collapse_person_to_{entity}"
            else:
                raise KeyError(
                    f"no categorical mapping from {native} to {entity} for {variable!r}"
                )
        else:
            mapped = simulation.calculate(variable, year, map_to=entity)
            values = _values(mapped)
            route = f"map_to_{entity}"

    if len(values) != len(table):
        raise ValueError(
            f"{variable!r} produced {len(values)} values for {len(table)} "
            f"{entity} rows."
        )
    if values.dtype.kind not in {"O", "U", "S"}:
        values = values.astype(float)
    return values, route


def _loss_relief_measure_from_engine(
    frame: Any, simulation: Any, variable: str, year: int
) -> tuple[np.ndarray, str]:
    """A loss-relief measure from the engine's components at ``year``."""

    profit = compute_uk_measure_input(
        frame, simulation, "person", "self_employment_income", year
    )[0]
    relief = compute_uk_measure_input(
        frame, simulation, "person", LOSS_RELIEF_COLUMN, year
    )[0]
    after = self_employment_income_after_loss_relief(profit, relief)
    if variable == UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF:
        return after, "engine_components:self_employment_income_less_loss_relief"
    total, _ = compute_uk_measure_input(
        frame, simulation, "person", "total_income", year
    )
    return (
        np.asarray(total, dtype=float) - np.asarray(profit, dtype=float) + after,
        "engine_components:total_income_with_profit_after_loss_relief",
    )


def compute_uc_paid_diagnostic_masks(
    frame: Any, simulation: Any, year: int
) -> dict[str, np.ndarray]:
    """Return out-of-fit Boolean masks in the frame's benefit-unit row order.

    The simulation must use that same row order, as for all UK measure inputs.
    Callers retain their own weights and source-ancestry aggregation. Model
    zero awards are labelled as such: they do not identify administrative open
    nil-payment claims. Family and child-count partitions are separate, matching
    the two available publisher cubes rather than inventing their joint.
    """
    uc = _values(simulation.calculate("universal_credit", year))
    if (
        uc.ndim != 1
        or len(uc) != len(frame.table("benunit"))
        or uc.dtype.kind not in "biuf"
        or not np.isfinite(uc).all()
        or (uc < 0).any()
    ):
        raise ValueError(
            "UC diagnostic awards must be finite nonnegative benunit values."
        )
    structural_family, reported_children = _uc_calibration_composition(
        frame, simulation, year
    )
    administrative_family = uc_administrative_family_type(
        frame, simulation, year, reported_children
    )
    own_qualifying_children = compute_uc_target_measure(
        frame, simulation, "uc_tcl_qualifying_child_count", year
    )
    entitled = uc_child_entitlement(frame, simulation, year)
    paid = uc > 0
    masks = {
        "paid.total": paid,
        "model.zero_award": uc == 0,
        "paid.child_entitled": paid & entitled,
        "paid.no_child_entitlement": paid & ~entitled,
        "paid.family_measure_disagreement": paid
        & (administrative_family != structural_family),
        "paid.child_count_measure_disagreement": paid
        & (reported_children != own_qualifying_children),
    }
    partitions = {
        "family": {
            category: administrative_family == category
            for category in (
                "SINGLE",
                "LONE_PARENT",
                "COUPLE_NO_CHILDREN",
                "COUPLE_WITH_CHILDREN",
                "UNKNOWN",
            )
        },
        "children": {
            **{str(count): reported_children == count for count in range(5)},
            "5_or_more": reported_children >= 5,
        },
    }
    for dimension, categories in partitions.items():
        for category, selected in categories.items():
            masks[f"paid.{dimension}.{category}"] = paid & selected
            masks[f"paid_child_entitled.{dimension}.{category}"] = (
                paid & entitled & selected
            )
            masks[f"paid_no_child_entitlement.{dimension}.{category}"] = (
                paid & ~entitled & selected
            )
    return masks


ENGINE_WEIGHT_SCALE_REASON = (
    "engine population representation: household weights scaled from the "
    "engine block's mass to the pool's so weight-share formulas (a national "
    "total allocated by x*w / sum(x*w)) see the pool's denominator; engine "
    "scratch only, discarded after resolution"
)


def _release_identifier_tables(
    tables: Mapping[str, pd.DataFrame], universe: Any | None
) -> dict[str, pd.DataFrame] | None:
    """The engine tables on the release's identifiers, or None if they already are.

    The build identifiers the release keeps beside them are not engine inputs
    and do not go to the scratch file.
    """

    whole = (
        None
        if universe is None
        else {entity: universe.table(entity) for entity in UK_BUILD_IDENTIFIER_COLUMNS}
    )
    released = uk_release_identifier_tables(
        {entity: tables[entity] for entity in UK_BUILD_IDENTIFIER_COLUMNS},
        universe=whole,
    )
    result = {}
    changed = False
    for entity, carrier in UK_BUILD_IDENTIFIER_COLUMNS.items():
        if carrier in tables[entity].columns:
            # Already on release identifiers (a loaded release): leave it.
            return None
        table = released[entity].drop(columns=[carrier])
        changed = changed or not table.equals(tables[entity])
        result[entity] = table
    return result if changed else None


def _engine_scratch_frame(
    frame: Any,
    *,
    engine_weight_scale: float | None = None,
    identifier_universe: Any | None = None,
) -> tuple[Any, tuple[str, ...]]:
    """The frame a scratch-mode engine loads, and the columns it leaves behind.

    The scratch H5 is an engine boundary like the release export, so the
    release export's reviewed drops (``UK_RELEASE_EXPORT_DROPPED_COLUMNS``)
    leave here too. Among them ``property_wealth`` is formula-owned in
    policyengine-uk: a persisted copy would override the engine's sum of the
    three persisted property components in the calibration year
    (microcosm#1106). The resolver's own frame keeps every column;
    frame-column measures read it.

    The nation-native alias codes of the derived geography layers
    (microcosm#931) are NA outside their own nation by design and are never
    engine inputs; the single-year dataset policyengine-uk loads refuses any
    NaN column, so they leave here exactly as they leave at the export
    boundary.

    ``engine_weight_scale`` multiplies the household weights the engine sees
    (and only those: the resolver's own frame keeps the block's true mass).
    A per-clone engine block carries a K-th of the pool, so a formula that
    allocates a national total by weighted share would hand every block the
    whole total; scaled to the pool's mass the block's weighted sums equal
    the pool's and the formula is exact for identical clone copies. The
    scaling is declared on the engine frame's mass log.

    The identifiers leave as the release's too (microcosm#1194,
    ``national_frame.uk_release_identifier_tables``): policyengine-uk stores
    them as 32-bit integers and keys a person's birth month on ``person_id``,
    so a solve measured on other identifiers fits a deal of birth dates the
    released file does not have. ``identifier_universe`` is the whole frame
    when this one is a per-clone block of it, so a block's rows take the
    identifiers they have in the release. The resolver's own frame keeps the
    build identifiers.

    A frame that needs none of this is returned as is.
    """

    trimmed: dict[str, pd.DataFrame] = {}
    for entity, columns in UK_RELEASE_EXPORT_DROPPED_COLUMNS.items():
        table = frame.table(entity)
        present = [column for column in columns if column in table.columns]
        if present:
            trimmed[entity] = table.drop(columns=present)
    household = trimmed.get("household", frame.table("household"))
    kept = without_uk_native_alias_columns(household)
    scale = None if engine_weight_scale is None else float(engine_weight_scale)
    if scale is not None and not (np.isfinite(scale) and scale > 0.0):
        raise ValueError("engine weight scale must be a positive finite factor.")
    renumbered = _release_identifier_tables(
        {
            **{
                entity: trimmed.get(entity, frame.table(entity))
                for entity in UK_BUILD_IDENTIFIER_COLUMNS
            },
            "household": kept,
        },
        identifier_universe,
    )
    if (
        not trimmed
        and kept is household
        and renumbered is None
        and (scale is None or scale == 1.0)
    ):
        return frame, ()
    dropped = tuple(
        column
        for entity in UK_RELEASE_EXPORT_DROPPED_COLUMNS
        for column in frame.table(entity).columns
        if entity in trimmed and column not in trimmed[entity].columns
    ) + tuple(c for c in household.columns if c not in kept.columns)
    tables = {name: frame.table(name) for name in frame.entities}
    tables.update(trimmed)
    tables["household"] = kept
    if renumbered is not None:
        tables.update(renumbered)
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    mass_log = frame.mass_log
    if scale is not None and scale != 1.0:
        if "household" not in weights:
            raise ValueError("engine weight scaling needs explicit household weights.")
        household_weights = weights["household"]
        scaled = Weights(
            np.asarray(household_weights.values, dtype=np.float64) * scale,
            kind=household_weights.kind,
        )
        mass_log = (
            *mass_log,
            MassChangeRecord(
                entity="household",
                old_total=float(household_weights.total),
                new_total=float(scaled.total),
                declared_factor=scale,
                reason=ENGINE_WEIGHT_SCALE_REASON,
            ),
        )
        weights["household"] = scaled
    engine_frame = Frame(
        {**tables, **{name: frame.link(name) for name in frame.links}},
        frame.schema,
        weights,
        frame.strata,
        mass_log=mass_log,
        metadata=frame.metadata,
    )
    return engine_frame, dropped


def engine_bus_pricing_parameters(tax_benefit_system: Any, year: int) -> dict[str, Any]:
    """policyengine-uk's bus boardings per trip and yield per boarding by area.

    Read at ``year`` in the resolver's own period form (the engine resolves
    a year to its 1 January instant) for every area the stage prices
    (``UK_ENGINE_BUS_AREA_BY_STAGE_AREA``); raises AttributeError when the
    engine carries no such parameters. An engine that moves the benchmark
    yield by the published fares index adds its ``benchmark_year`` and
    ``local_bus_fares_index_ratio``, the index in ``year`` over the benchmark
    year's, by the areas the fare receipts move by.
    """

    from microcosm.build.uk_runtime.bus_fare_pricing import (
        UK_ENGINE_BUS_AREA_BY_STAGE_AREA,
    )
    from microcosm.build.uk_runtime.ledger_targets import (
        UK_ENGINE_BUS_INDEX_AREA_BY_SERIES,
    )

    parameters = getattr(tax_benefit_system, "parameters", None)
    if parameters is None:
        raise AttributeError("the engine exposes no parameters")
    bus = parameters(str(int(year))).gov.dft.bus
    engine: dict[str, Any] = {
        name: {
            area: float(getattr(getattr(bus, name), area))
            for area in UK_ENGINE_BUS_AREA_BY_STAGE_AREA.values()
        }
        for name in ("boardings_per_trip", "yield_per_boarding")
    }
    # From policyengine-uk 2.145.0 (PolicyEngine/policyengine-uk#2546) the
    # benchmark yield moves with each area's fares index, read as the engine
    # reads it: the model year's blend over the benchmark year's.
    index = getattr(bus, "local_bus_fares_index", None)
    benchmark_year = getattr(bus, "benchmark_year", None)
    if index is not None and benchmark_year is not None:
        benchmark = parameters(
            f"{int(benchmark_year)}-01-01"
        ).gov.dft.bus.local_bus_fares_index
        engine["benchmark_year"] = int(benchmark_year)
        engine["local_bus_fares_index_ratio"] = {
            area: float(getattr(index, area)) / float(getattr(benchmark, area))
            for area in UK_ENGINE_BUS_INDEX_AREA_BY_SERIES.values()
        }
    return engine


class UKMeasureResolver:
    """B2 measure provider backed by a policyengine-uk Microsimulation."""

    def __init__(
        self,
        *,
        simulation_source: Path | None,
        scratch_dir: Path,
        year: int,
        frame: Any,
        microsimulation_factory: Any | None = None,
        engine_weight_scale: float | None = None,
        engine_identifier_universe: Any | None = None,
    ):
        self.year = int(year)
        self._engine_weight_scale = (
            None if engine_weight_scale is None else float(engine_weight_scale)
        )
        policyengine_uk = _policyengine_uk_module()
        factory = (
            microsimulation_factory
            if microsimulation_factory is not None
            else policyengine_uk.Microsimulation
        )
        if simulation_source is None:
            if frame is None:
                raise ValueError("scratch-mode UKMeasureResolver requires a frame.")
            scratch_dir.mkdir(parents=True, exist_ok=True)
            source_path = scratch_dir / "simulation-input.h5"
            engine_frame, dropped = _engine_scratch_frame(
                frame,
                engine_weight_scale=self._engine_weight_scale,
                identifier_universe=engine_identifier_universe,
            )
            write_uk_national_frame(engine_frame, source_path)
            mode = "scratch_frame_export"
        else:
            if self._engine_weight_scale is not None:
                raise ValueError(
                    "engine weight scaling needs scratch-mode resolution; a direct "
                    "H5 source carries its own weights."
                )
            if engine_identifier_universe is not None:
                raise ValueError(
                    "an identifier universe needs scratch-mode resolution; a direct "
                    "H5 source carries its own identifiers."
                )
            dropped = ()
            source_path = Path(simulation_source)
            mode = "direct_h5"
            if frame is None:
                frame, _provenance = load_uk_national_frame(source_path)
        self.frame = frame
        self._engine_scratch_dropped_columns = tuple(dropped)
        self._factory = factory
        self._source_path = source_path
        self._counterfactuals: dict[tuple[str, str | None], Any] = {}
        self._counterfactual_measures: dict[str, dict[str, Any]] = {}
        # The domestic energy measures, computed together on first read.
        self._household_energy_cache: dict[str, np.ndarray] | None = None
        reserved = {
            str(name)
            for name in frame.table("person")
            if _cgt_model_measure(str(name), self.year) is not None
        }
        if reserved:
            raise ValueError(
                "CGT calibration measures must not be persisted as source inputs: "
                f"{sorted(reserved)}"
            )
        persisted_boardings = sorted(
            set(_BUS_BOARDING_MEASURES) & set(map(str, frame.table("household")))
        )
        if persisted_boardings:
            raise ValueError(
                "Bus boarding measures must not be persisted as source inputs: "
                f"{persisted_boardings}"
            )
        persisted_high_value = sorted(
            set(_HIGH_VALUE_HOME_MEASURES) & set(map(str, frame.table("household")))
        )
        if persisted_high_value:
            raise ValueError(
                "High-value home measures must not be persisted as source inputs: "
                f"{persisted_high_value}"
            )
        persisted_spring_term = sorted(
            set(DFE_SPRING_TERM_MEASURES) & set(map(str, frame.table("person")))
        )
        if persisted_spring_term:
            raise ValueError(
                "DfE spring-term measures must not be persisted as source inputs: "
                f"{persisted_spring_term}"
            )
        persisted_centax = sorted(
            set(CENTAX_PARENT_MEASURES) & set(map(str, frame.table("person")))
        )
        if persisted_centax:
            raise ValueError(
                "CenTax parent measures must not be persisted as source inputs: "
                f"{persisted_centax}"
            )
        self.simulation = factory(dataset=str(source_path))
        validate_uc_claimant_input(self.simulation, frame.table("person"), self.year)
        self.contract_targets = _uk_contract_targets()
        bound_cgt_periods = {}
        for target_id, target in self.contract_targets.items():
            binding = target["bindings"]["policyengine"]
            for name in _binding_variable_names(binding):
                cgt_measure = _cgt_model_measure(name, self.year)
                if cgt_measure is None:
                    continue
                model_variable, measure_year = cgt_measure
                if int(binding.get("measurement_period", self.year)) != measure_year:
                    raise ValueError(
                        f"CGT amount/threshold period mismatch: {target_id}"
                    )
                bound_cgt_periods[name] = {
                    "model_variable": model_variable,
                    "measurement_period": measure_year,
                }
        self._receipt = {
            "mode": mode,
            "source_path": str(source_path),
            "policyengine_uk_version": _policyengine_uk_version(policyengine_uk),
            "cgt_period_contract": {
                "version": "uk-cgt-measurement-v3",
                "input_period": getattr(frame, "metadata", {}).get("time_period"),
                "calibration_period": self.year,
                "default_engine_period": self.year,
                "bound_measurements": bound_cgt_periods,
                "dated_measures": {
                    "naming": (
                        "cgt_<disposal_year>_<gains|tax|badr_gains|taxable_income>"
                    ),
                    "derived_measures": {
                        _CGT_TAXABLE_INCOME: _CGT_TAXABLE_INCOME_DEFINITION
                    },
                    "scanned_binding_keys": list(_BINDING_VARIABLE_KEYS),
                    "period": "explicit_disposal_year_in_variable_name",
                    "policy_threshold_period": "binding.measurement_period",
                },
                "model_variables": dict(_CGT_CALIBRATION_VARIABLES),
                "gains_basis": "after_losses_before_annual_exempt_amount",
                "losses_treatment": "already_in_net_gains_no_second_deduction",
                "population": "individuals",
                "taxpayer_proxy": "engine_year_net_gains_above_engine_year_aea",
                "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
                "input_mutation": False,
            },
        }

        self._uc_tcl_measures_used: set[str] = set()
        self._uc_paid_measures_used: set[str] = set()
        self._annual_income_measures_used: set[str] = set()
        self._annual_income_fallback: str | None = None
        self._annual_income_simulation_cache: Any | None = None
        self._annual_income_factors: dict[str, float] = {}

    def use_current_income_for_annual_measures(self, *, reason: str) -> None:
        """Let a frame without annual income descriptors read current values.

        Only a file built before the descriptors existed (the incumbent a
        candidate is scored against) takes this: its annual measures are its
        current ones, a share of 1 and no other spells. A frame that carries
        the descriptors still reads them, and any other frame without them
        refuses every annual measure.
        """

        if not str(reason).strip():
            raise ValueError("the current-income fallback needs a reason.")
        self._annual_income_fallback = str(reason)

    def _carries_annual_income_descriptors(self) -> bool:
        person = self.frame.table("person")
        present = [
            column for column in FRS_ANNUAL_INCOME_OUTPUT_COLUMNS if column in person
        ]
        if present and len(present) != len(FRS_ANNUAL_INCOME_OUTPUT_COLUMNS):
            missing = sorted(set(FRS_ANNUAL_INCOME_OUTPUT_COLUMNS) - set(present))
            raise ValueError(
                f"The frame carries only some annual income descriptors; missing "
                f"{missing}."
            )
        return bool(present)

    def validate_period(self, period: int | str) -> None:
        """Refuse thresholds and simulated inputs from different periods."""
        if str(period) != str(self.year):
            raise ValueError(
                f"UK measurement period {self.year} does not match target period "
                f"{period}."
            )

    def knows(self, entity: str, variable: str) -> bool:
        """Whether a route in :func:`compute_uk_measure_input` reaches here.

        The resolution loop turns a False into a named refusal; anything this
        answers True to must actually compute, or the operator gets a wrapped
        provider exception instead of the fence's message.
        """

        if (
            variable
            in _UC_CALIBRATION_VARIABLES
            | UC_TARGET_VARIABLES
            | UC_PAID_TARGET_VARIABLES
        ):
            return entity == "benunit"
        if variable in _BUS_BOARDING_MEASURES or variable in _HIGH_VALUE_HOME_MEASURES:
            return entity == "household"
        if variable in DFE_SPRING_TERM_MEASURES:
            return entity == "person" and all(
                self.knows("person", name) for name in DFE_SPRING_TERM_ENGINE_INPUTS
            )
        if variable == CENTAX_PARENT_OF_ELIGIBLE_3_4:
            return entity == "person" and all(
                self.knows("person", name) for name in CENTAX_PARENT_ENGINE_INPUTS
            )
        if variable == CENTAX_EMPLOYEE_MAIN_SOURCE:
            return entity == "person" and all(
                self.knows("person", name) for name in CENTAX_EMPLOYEE_SOURCE_INPUTS
            )
        cgt_measure = _cgt_model_measure(variable, self.year)
        if cgt_measure is not None:
            return entity == "person" and self.knows(entity, cgt_measure[0])
        if variable == _CGT_TAXABLE_INCOME:
            return entity == "person" and all(
                self.knows("person", name) for name in _CGT_TAXABLE_INCOME_COMPONENTS
            )
        if variable in _LOSS_RELIEF_MEASURES:
            return entity == "person" and all(
                self.knows("person", name) for name in _LOSS_RELIEF_MEASURE_COMPONENTS
            )
        if variable in UK_HOUSEHOLD_ENERGY_MEASURES:
            return entity == "household"
        if variable in UK_ANNUAL_INCOME_MEASURES:
            readable = (
                self._carries_annual_income_descriptors()
                or self._annual_income_fallback is not None
            )
            return readable and self.knows(entity, UK_ANNUAL_INCOME_MEASURES[variable])
        definition = self.simulation.tax_benefit_system.variables.get(variable)
        if definition is None or entity not in _ENTITY_ID:
            return False
        native = definition.entity.key
        if native == entity:
            return True
        if "person" in (native, entity):
            # Person-to-group and group-to-person are covered by the
            # broadcast, any-collapse, and map_to routes alike.
            return True
        # Group to group has only the numeric map_to route: a categorical or
        # boolean variable has no mapping between benunit and household and
        # refuses here rather than deep inside the provider.
        return getattr(definition, "value_type", None) in (int, float)

    def entity_for(self, variable: str) -> str | None:
        if variable in _BUS_BOARDING_MEASURES or variable in _HIGH_VALUE_HOME_MEASURES:
            return "household"
        if _cgt_model_measure(variable, self.year) is not None:
            return "person"
        if (
            variable in _LOSS_RELIEF_MEASURES
            or variable in DFE_SPRING_TERM_MEASURES
            or variable in CENTAX_PARENT_MEASURES
        ):
            return "person"
        if variable in UK_HOUSEHOLD_ENERGY_MEASURES:
            return "household"
        if variable in UK_ANNUAL_INCOME_MEASURES:
            return self.entity_for(UK_ANNUAL_INCOME_MEASURES[variable])
        if (
            variable
            in _UC_CALIBRATION_VARIABLES
            | UC_TARGET_VARIABLES
            | UC_PAID_TARGET_VARIABLES
        ):
            return "benunit"
        definition = self.simulation.tax_benefit_system.variables.get(variable)
        if definition is None:
            return None
        return str(definition.entity.key)

    def compute(self, entity: str, variable: str) -> tuple[np.ndarray, str]:
        if variable in UK_ANNUAL_INCOME_MEASURES:
            return self._annual_income_measure(entity, variable)
        if variable == CENTAX_EMPLOYEE_MAIN_SOURCE:
            return self._employment_main_income_source(entity)
        if variable in UK_HOUSEHOLD_ENERGY_MEASURES:
            return self._household_energy(entity, variable)
        result = compute_uk_measure_input(
            self.frame, self.simulation, entity, variable, self.year
        )
        if variable.startswith("uc_tcl_") and variable in UC_TARGET_VARIABLES:
            self._uc_tcl_measures_used.add(variable)
        if variable in UC_PAID_TARGET_VARIABLES:
            if not hasattr(self, "_uc_paid_measures_used"):
                self._uc_paid_measures_used = set()
            self._uc_paid_measures_used.add(variable)
        return result

    def counterfactual_delta(
        self, binding: Mapping[str, Any], period: int | str
    ) -> tuple[np.ndarray, str]:
        """A binding's per-row output delta under its input substitution.

        One extra simulation per (zeroed input, folded-into) pair, from the same
        source at the measurement year: the zeroed input is set to zero and its
        amount added to the folded-into input (salary sacrifice returned to
        pay, uk-data's recipe), and the output's delta against this resolver's
        baseline comes back for the binding's entity (microcosm#1069 c11).

        A binding that declares ``allocation`` splits the delta by HMRC rate
        category instead (microcosm#1095, uk-data#533): its output is the rise
        in tax on the declared taxable income under each person's schedule,
        computed once per substitution pair and cached. The old adjusted-net-
        income ``band`` test stays refused: it compared after-allowance
        thresholds and ignored the Scottish bands.
        """

        self.validate_period(period)
        if binding.get("band"):
            raise ValueError(
                "banded counterfactual measures are refused: the band test "
                "compares adjusted net income with after-allowance thresholds "
                "and ignores the Scottish bands; declare a rate-class "
                "allocation instead (microcosm#1095)"
            )
        if binding.get("kind") != "input_substitution_counterfactual":
            raise ValueError(f"unsupported counterfactual kind {binding.get('kind')!r}")
        zeroed = str(binding["zeroed_input"])
        folded = binding.get("folded_into")
        folded = None if folded is None else str(folded)
        output = str(binding["output_variable"])
        entity = str(binding.get("from_entity") or "person")
        direction = str(binding.get("output_delta", "counterfactual_minus_baseline"))
        if direction not in (
            "counterfactual_minus_baseline",
            "baseline_minus_counterfactual",
        ):
            raise ValueError(f"unsupported output_delta {direction!r}")
        metric = str(binding.get("metric_name") or output)
        allocation = binding.get("allocation")
        if allocation is not None:
            rate_class = _declared_rate_class(allocation, output, direction, entity)
            delta = self._rate_class_relief(zeroed, folded)[rate_class]
            self._counterfactual_measures[metric] = {
                "entity": entity,
                "zeroed_input": zeroed,
                "folded_into": folded,
                "output_variable": output,
                "output_delta": direction,
                "allocation": str(allocation["kind"]),
                "rate_class": rate_class,
                "rows_nonzero": int(np.count_nonzero(delta)),
            }
            route = (
                f"policyengine-uk counterfactual: {zeroed} set to zero"
                + (f" and folded into {folded}" if folded else "")
                + f"; rise in tax on {entity}.{output} at the {rate_class} rate"
            )
            return delta, route
        counterfactual = self._counterfactual_simulation(zeroed, folded)
        baseline_values, _ = compute_uk_measure_input(
            self.frame, self.simulation, entity, output, self.year
        )
        counterfactual_values, _ = compute_uk_measure_input(
            self.frame, counterfactual, entity, output, self.year
        )
        delta = np.asarray(counterfactual_values, dtype=float) - np.asarray(
            baseline_values, dtype=float
        )
        if direction == "baseline_minus_counterfactual":
            delta = -delta
        self._counterfactual_measures[metric] = {
            "entity": entity,
            "zeroed_input": zeroed,
            "folded_into": folded,
            "output_variable": output,
            "output_delta": direction,
            "rows_nonzero": int(np.count_nonzero(delta)),
        }
        route = (
            f"policyengine-uk counterfactual: {zeroed} set to zero"
            + (f" and folded into {folded}" if folded else "")
            + f"; {entity}.{output} {direction}"
        )
        return delta, route

    def _rate_class_relief(
        self, zeroed: str, folded: str | None
    ) -> Mapping[str, np.ndarray]:
        """Each person's rise in tax on earned income by rate class, cached."""

        cache = getattr(self, "_rate_class_reliefs", None)
        if cache is None:
            cache = self._rate_class_reliefs = {}
            self._rate_class_receipts = {}
        key = (zeroed, folded)
        if key in cache:
            return cache[key]
        counterfactual = self._counterfactual_simulation(zeroed, folded)
        income = str(UK_RATE_CLASS_ALLOCATION["taxable_income"])
        flag = str(UK_RATE_CLASS_ALLOCATION["schedule_flag"])
        baseline_income, _ = compute_uk_measure_input(
            self.frame, self.simulation, "person", income, self.year
        )
        counterfactual_income, _ = compute_uk_measure_input(
            self.frame, counterfactual, "person", income, self.year
        )
        scottish, _ = compute_uk_measure_input(
            self.frame, self.simulation, "person", flag, self.year
        )
        # A donor's grossed-up Gift Aid extends their bands; returning the
        # sacrifice to pay does not change the gift.
        gross_gift, _ = compute_uk_measure_input(
            self.frame,
            self.simulation,
            "person",
            str(UK_RATE_CLASS_ALLOCATION["gift_aid_extends_bands"]),
            self.year,
        )
        # The simulation's own parameters at the measurement year are its
        # fiscal-year values, so the Scottish bands are the ones in force from
        # 6 April, as the engine taxes them.
        rates = self.simulation.tax_benefit_system.parameters(self.year)
        schedules = {}
        for name, path in dict(UK_RATE_CLASS_ALLOCATION["schedules"]).items():
            node = rates
            for part in str(path).split("."):
                node = getattr(node, part)
            schedules[name] = (list(node.thresholds), list(node.rates))
        relief = relief_by_rate_class(
            baseline_income=np.asarray(baseline_income, dtype=float),
            counterfactual_income=np.asarray(counterfactual_income, dtype=float),
            scottish=np.asarray(scottish, dtype=bool),
            schedules=schedules,
            gross_gift=np.asarray(gross_gift, dtype=float),
        )
        # The allocation re-implements the engine's earned_income_tax, the
        # schedule applied to earned taxable income with a donor's bands
        # extended by their gift. Its classes must add up
        # to that variable's rise, or the schedule has moved away from the
        # brackets read here.
        engine_variable = UK_RATE_CLASS_ENGINE_VARIABLE
        baseline_earned, _ = compute_uk_measure_input(
            self.frame, self.simulation, "person", engine_variable, self.year
        )
        counterfactual_earned, _ = compute_uk_measure_input(
            self.frame, counterfactual, "person", engine_variable, self.year
        )
        earned_tax_change = np.asarray(counterfactual_earned, dtype=float) - np.asarray(
            baseline_earned, dtype=float
        )
        earned_gap = np.abs(earned_tax_change - sum(relief.values()))
        # GBP 1 plus a few float32 steps at the person's own amounts: the
        # engine rounds a very large income to more than GBP 1.
        tolerance = rate_class_engine_tolerance(
            baseline_income,
            counterfactual_income,
            baseline_earned,
            counterfactual_earned,
        )
        beyond = earned_gap > tolerance
        if beyond.any():
            raise ValueError(
                f"The {UK_RATE_CLASS_ALLOCATION['kind']} allocation no longer adds "
                f"up to the engine's {engine_variable}: {int(beyond.sum())} "
                f"person(s) differ by more than GBP "
                f"{UK_RATE_CLASS_ENGINE_TOLERANCE_GBP:g} and "
                f"{UK_RATE_CLASS_ENGINE_FLOAT32_STEPS} float32 steps, by up to "
                f"GBP {float(earned_gap[beyond].max()):.2f}. "
                "Review the income tax schedule it reads."
            )
        baseline_tax, _ = compute_uk_measure_input(
            self.frame, self.simulation, "person", "income_tax", self.year
        )
        counterfactual_tax, _ = compute_uk_measure_input(
            self.frame, counterfactual, "person", "income_tax", self.year
        )
        income_tax_change = np.asarray(counterfactual_tax, dtype=float) - np.asarray(
            baseline_tax, dtype=float
        )
        classes_total = float(sum(np.sum(values) for values in relief.values()))
        self._rate_class_receipts[f"{zeroed}->{folded}"] = {
            "allocation": str(UK_RATE_CLASS_ALLOCATION["kind"]),
            "relief_by_class_unweighted_total": {
                name: float(np.sum(values)) for name, values in relief.items()
            },
            "engine_variable": engine_variable,
            "engine_variable_change_unweighted_total": float(np.sum(earned_tax_change)),
            "largest_gap_to_engine_variable": float(earned_gap.max())
            if earned_gap.size
            else 0.0,
            "income_tax_change_unweighted_total": float(np.sum(income_tax_change)),
            "gap_to_income_tax_change_unweighted": float(
                np.sum(income_tax_change) - classes_total
            ),
            "persons_with_gap": int(
                np.count_nonzero(
                    ~np.isclose(
                        income_tax_change,
                        sum(relief.values()),
                        rtol=0.0,
                        atol=0.01,
                    )
                )
            ),
        }
        cache[key] = relief
        return relief

    def _annual_income_measure(
        self, entity: str, variable: str
    ) -> tuple[np.ndarray, str]:
        """An annual measure: its current measure on the annual simulation."""

        current = UK_ANNUAL_INCOME_MEASURES[variable]
        if not self._carries_annual_income_descriptors():
            if self._annual_income_fallback is None:
                raise KeyError(
                    f"{variable} reads the annual income descriptors "
                    f"(frs_annual_income_spells), which this frame does not carry."
                )
            self._annual_income_measures_used.add(variable)
            values, route = compute_uk_measure_input(
                self.frame, self.simulation, entity, current, self.year
            )
            return values, f"current_values_without_annual_income_descriptors:{route}"
        simulation = self._annual_income_simulation()
        self._annual_income_measures_used.add(variable)
        if current in _LOSS_RELIEF_MEASURES:
            if entity != "person":
                raise KeyError(f"{variable} is person-only: {entity}")
            # Always the engine's components: the frame's stored columns are
            # the current year's, never the annual one.
            values, route = _loss_relief_measure_from_engine(
                self.frame, simulation, current, self.year
            )
        else:
            values, route = compute_uk_measure_input(
                self.frame, simulation, entity, current, self.year
            )
        return values, f"annual_income_simulation:{route}"

    def _employment_main_income_source(self, entity: str) -> tuple[np.ndarray, str]:
        """One where employment is the largest of the year's income sources."""

        if entity != "person":
            raise KeyError(f"{CENTAX_EMPLOYEE_MAIN_SOURCE} is person-only: {entity}")
        sources = {
            name: self.compute("person", name)[0]
            for name in CENTAX_EMPLOYEE_SOURCE_INPUTS
        }
        return (
            employment_main_income_source(sources),
            "annual_income_sources:" + ",".join(CENTAX_EMPLOYEE_SOURCE_INPUTS),
        )

    def _household_energy(self, entity: str, variable: str) -> tuple[np.ndarray, str]:
        """A household's domestic energy measure from its stored spend."""

        if entity != "household":
            raise KeyError(f"{variable} is household-only: {entity}")
        cached = self._household_energy_cache
        if cached is None:
            year = self._dataset_year()
            electricity = _values(
                self.simulation.calculate("electricity_consumption", year)
            ).astype(float)
            gas = _values(self.simulation.calculate("gas_consumption", year)).astype(
                float
            )
            raw_region = self.simulation.calculate("region", year)
            region = (
                np.asarray(raw_region.decode_to_str())
                if hasattr(raw_region, "decode_to_str")
                else _values(raw_region)
            )
            cached = self._household_energy_cache = {
                **household_energy_kwh(
                    electricity, gas, region=region, prices=household_energy_prices()
                ),
                _GAS_CONNECTED: (gas > 0).astype(float),
            }
        values = np.asarray(cached[UK_HOUSEHOLD_ENERGY_MEASURES[variable]], dtype=float)
        return values, "stored_spend_at_dataset_period:household_energy"

    def _dataset_year(self) -> int:
        metadata = getattr(self.frame, "metadata", None) or {}
        period = metadata.get(UK_TIME_PERIOD_METADATA_KEY)
        if isinstance(period, str) and period.strip():
            return int(period)
        return self.year

    def _annual_income_simulation(self) -> Any:
        """The simulation of each person's year, built once from the same source.

        The engine projects a dataset's inputs to its later years as it loads
        it, so the year's inputs are set at the measurement year: pay, profit,
        private pensions and the contributory benefits to the engine's current
        amount times its annual share, plus the other spells carried from the
        dataset's year by the factor the engine carries the current amount by.
        The State Pension is set to the share of it paid in the fiscal year,
        and the earnings-tested carer benefits to their baseline values.
        Pension contributions and salary sacrifice keep their current values:
        for a part-year entrant they can be large against the year's pay,
        which lowers the year's income tax a little.
        """

        if self._annual_income_simulation_cache is not None:
            return self._annual_income_simulation_cache
        person = self.frame.table("person")
        simulation = self._factory(dataset=str(self._source_path))
        annual_inputs: dict[str, np.ndarray] = {}
        for name, share_column, other_column in _ANNUAL_INCOME_INPUTS:
            current = _values(self.simulation.calculate(name, self.year)).astype(float)
            share = person[share_column].to_numpy(dtype=float)
            other = (
                person[other_column].to_numpy(dtype=float)
                if other_column is not None
                else np.zeros(len(person))
            )
            if current.shape != share.shape:
                raise ValueError(
                    f"{name} must align with the frame's person table to read "
                    "its annual income descriptors."
                )
            factor = self._engine_projection_factor(name) if other.any() else 1.0
            annual = current * share + other * factor
            if not np.isfinite(annual).all():
                raise ValueError(f"The annual {name} must be finite.")
            simulation.set_input(name, self.year, annual)
            annual_inputs[name] = annual
        months = _values(
            self.simulation.calculate("months_since_state_pension_age", self.year)
        )
        pension = _values(self.simulation.calculate("state_pension", self.year))
        annual_pension = pension.astype(float) * state_pension_annual_share(months)
        simulation.set_input("state_pension", self.year, annual_pension)
        annual_inputs["state_pension"] = annual_pension
        for name in _ANNUAL_INCOME_HELD_VARIABLES:
            held = _values(self.simulation.calculate(name, self.year)).astype(float)
            simulation.set_input(name, self.year, held)
            annual_inputs[name] = held
        for name, expected in annual_inputs.items():
            read = _values(simulation.calculate(name, self.year)).astype(float)
            tolerance = 1e-6 * np.maximum(np.abs(expected), 1.0) + 0.01
            if (
                read.shape != expected.shape
                or (np.abs(read - expected) > tolerance).any()
            ):
                raise ValueError(
                    f"The annual simulation does not read back the {name} set "
                    f"for {self.year}."
                )
        self._annual_income_simulation_cache = simulation
        return simulation

    def _engine_projection_factor(self, name: str) -> float:
        """The factor the engine carries ``name`` by from the dataset's year.

        Read off the engine's own projection, so the other spells grow by the
        series the current amount grows by; a projection that is not one
        factor for every row refuses, since no single factor then carries the
        other spells as the engine carries pay.
        """

        dataset_year = self._dataset_year()
        then = _values(self.simulation.calculate(name, dataset_year)).astype(float)
        now = _values(self.simulation.calculate(name, self.year)).astype(float)
        carried = np.abs(then) > 0
        if not carried.any():
            raise ValueError(
                f"No row carries {name} in {dataset_year}, so the engine's factor "
                f"to {self.year} for the other spells cannot be read."
            )
        factors = now[carried] / then[carried]
        factor = float(np.median(factors))
        if (np.abs(factors - factor) > 1e-5 * max(abs(factor), 1.0)).any():
            raise ValueError(
                f"The engine does not carry {name} from {dataset_year} to "
                f"{self.year} by one factor, so the other spells cannot follow it."
            )
        self._annual_income_factors[name] = factor
        return factor

    def _counterfactual_simulation(self, zeroed: str, folded: str | None) -> Any:
        key = (zeroed, folded)
        if key in self._counterfactuals:
            return self._counterfactuals[key]
        simulation = self._factory(dataset=str(self._source_path))
        amount = np.asarray(self.simulation.calculate(zeroed, self.year), dtype=float)
        if not np.isfinite(amount).all():
            raise ValueError(f"{zeroed} must be finite to substitute it.")
        if folded is not None:
            base = np.asarray(self.simulation.calculate(folded, self.year), dtype=float)
            if base.shape != amount.shape:
                raise ValueError(
                    f"{zeroed} and {folded} must share an entity to fold one into the other."
                )
            simulation.set_input(folded, self.year, base + amount)
        simulation.set_input(zeroed, self.year, np.zeros_like(amount))
        self._counterfactuals[key] = simulation
        return simulation

    def _bus_engine_parity(self) -> dict[str, Any]:
        """The engine's bus pricing against the lcfs stage's, computed once.

        Report only (microcosm#1207): the engine prices bus fares from the
        trips the dataset carries at its own boardings per trip and yield per
        boarding, so the receipt records how far those stand from the ones the
        stage priced at. An engine without them records why it has none.
        """

        cached = getattr(self, "_bus_parity", None)
        if cached is not None:
            return cached
        from microcosm.build.country_spec import load_country_spec
        from microcosm.build.uk_runtime.bus_fare_pricing import (
            bus_engine_parity,
            bus_fare_prices,
        )
        from microcosm.build.uk_runtime.lcfs_consumption import (
            UK_LCFS_VENDORED_RESOURCES,
            bus_pricing_operation,
        )

        declared = bus_pricing_operation(
            load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
        )
        try:
            if declared is None:
                raise AttributeError("the lcfs stage declares no bus pricing")
            engine = engine_bus_pricing_parameters(
                getattr(self.simulation, "tax_benefit_system", None), self.year
            )
        except AttributeError as error:
            parity: dict[str, Any] = {
                "report_only": True,
                "available": False,
                "reason": str(error),
            }
        else:
            from microcosm.build.uk_runtime.ledger_targets import (
                uk_bus_fare_alignment_factors,
            )

            parity = bus_engine_parity(
                bus_fare_prices(declared, allowed_resources=UK_LCFS_VENDORED_RESOURCES),
                engine,
                alignment_factors=(
                    uk_bus_fare_alignment_factors(
                        int(engine["benchmark_year"]), int(self.year)
                    )
                    if "benchmark_year" in engine
                    else None
                ),
            )
        self._bus_parity = parity
        return parity

    def receipt(self) -> dict[str, Any]:
        receipt = dict(self._receipt)
        receipt["bus_engine_parity"] = self._bus_engine_parity()
        if getattr(self, "_engine_weight_scale", None) is not None:
            receipt["engine_weight_scale"] = float(self._engine_weight_scale)
        if getattr(self, "_engine_scratch_dropped_columns", ()):
            receipt["engine_scratch_dropped_columns"] = list(
                self._engine_scratch_dropped_columns
            )
        counterfactual_measures = getattr(self, "_counterfactual_measures", None)
        if counterfactual_measures:
            receipt["counterfactual_measures"] = {
                name: dict(entry)
                for name, entry in sorted(counterfactual_measures.items())
            }
        rate_class_receipts = getattr(self, "_rate_class_receipts", None)
        if rate_class_receipts:
            receipt["rate_class_reliefs"] = {
                name: dict(entry) for name, entry in sorted(rate_class_receipts.items())
            }
        if self._uc_tcl_measures_used:
            receipt["uc_tcl_comparison_contract"] = {
                **uc_tcl_comparison_contract(self.year),
                "computed_measures": sorted(self._uc_tcl_measures_used),
            }
        if getattr(self, "_uc_paid_measures_used", None):
            receipt["uc_paid_comparison_contract"] = {
                **uc_paid_comparison_contract(self.year),
                "computed_measures": sorted(self._uc_paid_measures_used),
            }
        if getattr(self, "_annual_income_measures_used", None):
            receipt["annual_income_measures"] = self._annual_income_receipt()
        return receipt

    def _annual_income_receipt(self) -> dict[str, Any]:
        measures = {
            name: UK_ANNUAL_INCOME_MEASURES[name]
            for name in sorted(self._annual_income_measures_used)
        }
        if not self._carries_annual_income_descriptors():
            return {
                "computed_measures": measures,
                "route": "current_values_without_annual_income_descriptors",
                "reason": self._annual_income_fallback,
            }
        return {
            "computed_measures": measures,
            "route": "annual_income_simulation",
            "descriptors": list(FRS_ANNUAL_INCOME_OUTPUT_COLUMNS),
            "dataset_year": self._dataset_year(),
            "measurement_year": self.year,
            "inputs": {
                name: {"annual_share": share, "other_spells": other}
                for name, share, other in _ANNUAL_INCOME_INPUTS
            },
            "other_spells_engine_factors": dict(
                sorted(self._annual_income_factors.items())
            ),
            "state_pension": (
                "state_pension at the measurement year times the share of the "
                "fiscal year paid: (months_since_state_pension_age + 6) / 12 "
                "within six months of the year's middle, else 1"
            ),
            "held_at_baseline": list(_ANNUAL_INCOME_HELD_VARIABLES),
        }


#: Every field a reviewed measure exclusion must carry (the
#: ``weighted_integrity`` reviewed-exclusion record shape, microcosm#757 /
#: the #743-audit adjudication): an exclusion narrows the calibrated target
#: surface, so it names who approved the narrowing, under which adjudication,
#: and for how long — nothing lapses silently and nothing lives forever.
_UK_MEASURE_EXCLUSION_FIELDS = (
    "name",
    "reason",
    "tracking",
    "approved_by",
    "adjudication",
    "approved_on",
    "expires_on",
)


def load_uk_calibration_measure_exclusions(
    path: Path | None = None,
) -> tuple[dict[str, str], ...]:
    """Load and validate the reviewed UK calibration-measure exclusions."""

    if path is None:
        text = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath("calibration_measure_exclusions.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    # Schema 4 carries the signed zero-support floor (microcosm#1095 U3) in
    # scopes of one unit each, which
    # ``support_floor.load_uk_support_floor_declaration`` reads; schema 3's
    # single-scope floor is retired with it.
    schema_version = payload.get("schema_version")
    allowed_top = {"schema_version", "exclusions"}
    if schema_version == 4:
        allowed_top.add("support_floor")
    unknown = sorted(set(payload) - allowed_top)
    if unknown:
        raise ValueError(f"unknown top-level exclusion key(s): {unknown}")
    if schema_version not in (2, 4):
        raise ValueError(
            "UK calibration measure exclusions schema_version must be 2 or 4."
        )
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("UK calibration measure exclusions must contain a list.")
    seen: set[str] = set()
    loaded: list[dict[str, str]] = []
    for entry in exclusions:
        if not isinstance(entry, dict):
            raise ValueError(
                "UK calibration measure exclusion entries must be objects."
            )
        unknown_fields = sorted(set(entry) - set(_UK_MEASURE_EXCLUSION_FIELDS))
        if unknown_fields:
            raise ValueError(
                "unknown UK calibration measure exclusion field(s) "
                f"{unknown_fields} on {entry.get('name')!r}."
            )
        record = {
            field: str(entry.get(field, "")) for field in _UK_MEASURE_EXCLUSION_FIELDS
        }
        name = record["name"]
        if name in seen:
            raise ValueError(f"duplicate UK calibration measure exclusion {name!r}.")
        for field in _UK_MEASURE_EXCLUSION_FIELDS:
            if not record[field].strip():
                raise ValueError(
                    f"UK calibration measure exclusion {name!r} has empty {field}; "
                    "a narrowed target surface names why, where it is being "
                    "resolved, who approved it, and for how long."
                )
        for field in ("approved_on", "expires_on"):
            try:
                parsed = date.fromisoformat(record[field])
            except ValueError:
                parsed = None
            if parsed is None or parsed.isoformat() != record[field]:
                raise ValueError(
                    f"UK calibration measure exclusion {name!r} {field} must be "
                    f"canonical ISO (YYYY-MM-DD), got {record[field]!r}."
                )
        if record["expires_on"] <= record["approved_on"]:
            raise ValueError(
                f"UK calibration measure exclusion {name!r} expires_on must be "
                "after approved_on."
            )
        seen.add(name)
        loaded.append(record)
    return tuple(loaded)


def apply_uk_calibration_measure_exclusions(
    registry: TargetRegistry,
    exclusions: tuple[dict[str, str], ...],
    *,
    now: date | None = None,
) -> tuple[TargetRegistry, dict[str, dict[str, str]]]:
    """Remove reviewed excluded references from a UK target registry.

    The receipt carries every field the register declares, tracking included:
    an exclusion narrows the calibrated target surface, so the run evidence
    must say where each narrowing is being resolved, not only why. The window
    is enforced at apply time — outside ``approved_on``..``expires_on`` the
    run fails with a correct-or-renew message rather than the narrowing
    lapsing silently or living forever.
    """

    evaluated_on = exclusion_evaluation_date(now)
    for entry in exclusions:
        approved = date.fromisoformat(entry["approved_on"])
        expires = date.fromisoformat(entry["expires_on"])
        if not approved <= evaluated_on <= expires:
            raise ValueError(
                f"UK calibration measure exclusion {entry['name']!r} is outside "
                f"its reviewed window ({entry['approved_on']}..{entry['expires_on']}, "
                f"evaluated {evaluated_on.isoformat()}): correct the underlying "
                f"gap ({entry['tracking']}) or renew the adjudication with a new "
                "approval and expiry."
            )
        if (expires - evaluated_on).days <= 7:
            warnings.warn(
                f"UK calibration measure exclusion {entry['name']!r} expires "
                f"{entry['expires_on']} — within one week of the evaluation "
                f"date {evaluated_on.isoformat()}; correct the underlying gap "
                f"({entry['tracking']}) or renew the adjudication before the "
                "run fails on expiry.",
                UserWarning,
                stacklevel=2,
            )
    declared = {entry["name"]: entry for entry in exclusions}
    matched = {spec.name for spec in registry.specs if spec.name in declared}
    stale = sorted(set(declared) - matched)
    if stale:
        raise ValueError(
            "UK calibration measure exclusion matched zero registry specs: "
            + ", ".join(stale)
        )
    kept = [spec for spec in registry.specs if spec.name not in declared]
    receipt = {
        name: {
            field: declared[name][field]
            for field in _UK_MEASURE_EXCLUSION_FIELDS
            if field != "name"
        }
        for name in sorted(matched)
    }
    for record in receipt.values():
        record["evaluated_on"] = evaluated_on.isoformat()
    return TargetRegistry(kept, country=registry.country), receipt


def _values(raw: Any) -> np.ndarray:
    return np.asarray(raw.values if hasattr(raw, "values") else raw)


def _policyengine_uk_version(module: Any) -> str:
    """The engine version: the module's own ``__version__`` when it declares
    one (test doubles do), else the installed distribution metadata (the real
    module exposes none), else ``unknown``."""

    declared = getattr(module, "__version__", None)
    if declared:
        return str(declared)
    try:
        from importlib.metadata import version

        return str(version("policyengine-uk"))
    except Exception:  # pragma: no cover - metadata absent in odd envs
        return "unknown"


def _policyengine_uk_module() -> Any:
    import policyengine_uk

    return policyengine_uk


def _uk_contract_targets() -> dict[str, Any]:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    contract = json.loads(payload)
    from microcosm.build.uk_runtime.ledger_targets import (
        _warn_on_undeclared_geography,
    )

    _warn_on_undeclared_geography(contract["targets"])
    national_geography_levels = {"country", "region"}
    return {
        target["target_id"]: target
        for target in contract["targets"]
        if set(target.get("geography_levels") or ()) <= national_geography_levels
    }


def _declared_rate_class(
    allocation: object, output: str, direction: str, entity: str
) -> str:
    """The rate class a binding's allocation names, refused unless declared."""

    if not isinstance(allocation, Mapping):
        raise ValueError("a counterfactual allocation must be a mapping")
    rate_class = allocation.get("rate_class")
    expected = {**UK_RATE_CLASS_ALLOCATION, "rate_class": rate_class}
    if (
        rate_class not in UK_RATE_CLASSES
        or dict(allocation) != expected
        or output != UK_RATE_CLASS_ALLOCATION["taxable_income"]
        or direction != "counterfactual_minus_baseline"
        or entity != "person"
    ):
        raise ValueError(
            "a rate-class allocation must declare "
            f"{dict(UK_RATE_CLASS_ALLOCATION)} with a rate_class in "
            f"{UK_RATE_CLASSES}, on person {UK_RATE_CLASS_ALLOCATION['taxable_income']} "
            f"counterfactual_minus_baseline; got {allocation!r} on {entity}.{output} "
            f"{direction}"
        )
    return str(rate_class)
