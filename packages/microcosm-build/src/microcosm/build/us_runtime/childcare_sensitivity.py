"""Assumption stress tests for noncalendar schedules; never source observations."""

from __future__ import annotations

import gc

import numpy as np

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.nsece_childcare import NSECEChildcareSource
from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL


def noncalendar_sensitivity_source(source, *, irregular_hours=True, day_shift=0):
    """Change only modeled bridge components, keeping measured regular hours.

    The selected donor's days shift by one, bounded by 1--7 and the minimum
    feasible days at 24 hours/day. Zero total hours always means zero days.
    These brackets are stress tests, not confidence intervals.
    """
    if day_shift not in (-1, 0, 1):
        raise ValueError("The sensitivity day shift must be -1, 0 or 1.")
    children = source.children.copy()
    rows = children.attendance_status.eq("summary_bridge")
    month, days, hours = US_CHILDCARE_ATTENDANCE_COLUMNS
    regular = children.loc[rows, "regular_hours_per_week"].to_numpy(dtype=float)
    irregular = children.loc[rows, "irregular_hours_per_week"].to_numpy(dtype=float)
    if not irregular_hours:
        irregular = np.zeros_like(irregular)
    total = regular + irregular
    attended = np.clip(children.loc[rows, days].to_numpy(dtype=float) + day_shift, 1, 7)
    attended = np.where(total > 0, np.maximum(attended, np.ceil(total / 24)), 0)
    children.loc[rows, days] = attended
    children.loc[rows, month] = np.floor(attended * 52 / 12 + 0.5)
    children.loc[rows, hours] = np.divide(
        total, attended, out=np.zeros_like(total), where=attended > 0
    )
    children.loc[rows, "ece_hours_per_week"] = total
    children.loc[rows, "irregular_hours_per_week"] = irregular
    return NSECEChildcareSource(
        children,
        source.weights,
        {
            **source.source_receipt,
            "transport_sensitivity": {
                "include_modeled_irregular_hours": irregular_hours,
                "modeled_day_shift": day_shift,
                "measured_regular_hours_preserved": True,
            },
        },
    )


def compare_childcare_scenarios(parent, scenarios, *, year=2026):
    """Yield state comparisons on one fixed population; vary only attendance."""
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    people = parent.table("person")
    young = people.age.between(0, 12).to_numpy()
    for name, values in scenarios.items():
        if np.shape(values) != (len(people), 3) or not np.isfinite(values[young]).all():
            raise ValueError(f"Invalid scenario attendance: {name}")
    households = parent.table("household").copy()
    households["household_weight"] = parent.weights_for("household").values
    for fips, group in households.groupby("state_fips", sort=True):
        state = US_STATE_NUMERIC_FIPS_TO_POSTAL[int(fips)]
        selected = people.person_household_id.isin(group.household_id)
        state_people = people.loc[selected].copy()
        tables = {"person": state_people, "household": group}
        for entity in parent.schema.group_entities:
            if entity != "household":
                table = parent.table(entity)
                tables[entity] = table.loc[
                    table[f"{entity}_id"].isin(state_people[f"person_{entity}_id"])
                ]
        result = {"state": state, "sample_households": len(group)}
        for name, attendance in {"baseline": None, **scenarios}.items():
            sim = Microsimulation(
                dataset=USSingleYearDataset(**tables, time_period=year)
            )
            if attendance is not None:
                for i, column in enumerate(US_CHILDCARE_ATTENDANCE_COLUMNS):
                    values = np.asarray(sim.calculate(column, year)).copy()
                    specified = attendance[selected, i]
                    known = np.isfinite(specified)
                    values[known] = specified[known]
                    sim.set_input(column, year, values)
            variable = f"{state.lower()}_child_care_subsidies"
            definition = sim.tax_benefit_system.variables[variable]
            amount = np.asarray(sim.calculate(variable, year), dtype=float)
            weights = np.asarray(
                sim.calculate(f"{definition.entity.key}_weight", year), dtype=float
            )
            if not np.isfinite(amount).all():
                raise ValueError(f"Nonfinite state benefit: {state}/{name}")
            result[name] = {
                "variable": variable,
                "entity": definition.entity.key,
                "positive_sample_units": int((amount > 0).sum()),
                "weighted_positive_units": float(weights[amount > 0].sum()),
                "annual_modeled_benefits": float(amount @ weights),
            }
            del sim
            gc.collect()
        yield result
