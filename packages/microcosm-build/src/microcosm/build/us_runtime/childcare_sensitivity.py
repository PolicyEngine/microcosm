"""Assumption stress tests for noncalendar schedules; never source observations."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    _ids,
    _validate_attendance,
)
from microcosm.build.us_runtime.nsece_childcare import NSECEChildcareSource
from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL


def paired_noncalendar_attendance(people, source, alternative):
    """Apply an assumption change to the same selected donor, preserving observations.

    Quantile-coupled retransfers can change donor identities when schedules change
    sort order. This contrast instead conditions on the realized donor assignment.
    It is an assumption diagnostic, not a replacement population or receipt.
    """
    columns = list(US_CHILDCARE_ATTENDANCE_COLUMNS)
    before = source.children.set_index("donor_id", drop=False).sort_index()
    after = alternative.children.set_index("donor_id", drop=False).sort_index()
    _ids(before, "donor_id", unique=True)
    _ids(after, "donor_id", unique=True)
    if not before.index.equals(after.index):
        raise ValueError("Paired sensitivity requires the same donor identities.")
    # These are the only reconstructed fields the scenario may change.
    modeled = [*columns, "ece_hours_per_week", "irregular_hours_per_week"]
    pd.testing.assert_frame_equal(
        before.drop(columns=modeled, errors="ignore"),
        after.drop(columns=modeled, errors="ignore"),
        check_exact=True,
    )
    fixed = before.attendance_status.ne("summary_bridge")
    pd.testing.assert_frame_equal(
        before.loc[fixed, columns], after.loc[fixed, columns], check_exact=True
    )
    _validate_attendance(before, complete=False)
    _validate_attendance(after, complete=False)
    _validate_attendance(people, complete=False)
    result = (
        people.reindex(columns=columns).to_numpy(dtype=float, na_value=np.nan).copy()
    )
    original = result.copy()
    labels = people.reindex(columns=[f"{c}_source" for c in columns]).astype("string")
    selected = labels.apply(lambda col: col.str.startswith("donor:", na=False))
    donor_ids = labels.where(selected).apply(lambda col: col.str.removeprefix("donor:"))
    if donor_ids.nunique(axis=1).gt(1).any():
        raise ValueError("Paired sensitivity found mixed donor lineage within a child.")
    row_donor = donor_ids.bfill(axis=1).iloc[:, 0]
    uses_donor = selected.any(axis=1).to_numpy()
    if (uses_donor & ~people.age.between(0, 12).to_numpy()).any():
        raise ValueError("Paired sensitivity found an out-of-domain donor assignment.")
    if (
        row_donor.loc[uses_donor].isna().any()
        or not row_donor.loc[uses_donor].isin(before.index).all()
    ):
        raise ValueError("Paired sensitivity cannot resolve a selected donor.")
    expected = before.reindex(row_donor).reindex(columns=columns).to_numpy(dtype=float)
    changed = after.reindex(row_donor).reindex(columns=columns).to_numpy(dtype=float)
    mask = selected.to_numpy(dtype=bool)
    if not np.array_equal(original[mask], expected[mask]):
        raise ValueError("Candidate attendance does not match its selected donor.")
    # Observed/clone-inherited constraints must remain compatible with the donor.
    constrained = uses_donor[:, None] & ~mask & np.isfinite(original)
    if not np.array_equal(original[constrained], changed[constrained]):
        raise ValueError("Paired scenario conflicts with observed attendance.")
    result[mask] = changed[mask]
    _validate_attendance(pd.DataFrame(result, columns=columns), complete=False)
    modified = np.any(~np.isclose(result, original, equal_nan=True), axis=1)
    bridge = row_donor.map(before.attendance_status).eq("summary_bridge").to_numpy()
    if (modified & ~bridge).any():
        raise ValueError("Paired scenario changed a measured-calendar assignment.")
    return result, {
        "donor_assigned_people": int(uses_donor.sum()),
        "bridge_assigned_people": int(bridge.sum()),
        "changed_people": int(modified.sum()),
        "measured_calendar_assignments_changed": 0,
        "donor_assignment": "held fixed and validated against candidate values",
    }


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
