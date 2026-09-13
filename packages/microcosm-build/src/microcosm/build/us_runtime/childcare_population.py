"""Harmonize measured ASEC parent relationships for child attendance candidates.

The source adapter resolves Census PEPAR1/PEPAR2 within physical households.
It never equates all adults with parents. The downstream donor kernel consumes
only normalized age, region and parent-work fields, independent of source spine.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare_attendance import (
    childcare_attendance_contract,
    childcare_income_band,
)
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
)
from microcosm.calibrate.geography_constants import (
    US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION,
)
from microcosm.frame import US_SCHEMA, Frame


def harmonize_asec_childcare_predictors(
    frame: Frame, *, source_cache: str | Path | None = None
) -> Frame:
    """Normalize parents of *any* under-13 child, matching NSECE's household unit.

    Census documents PEPAR1/PEPAR2 as parent line numbers and A_LINENO as the
    unique person line number. Nonpositive pointers mean no resident parent;
    positive pointers must resolve in the same household. Current-week work is
    measured by hours_worked_last_week, not annual earnings or usual hours.
    https://api.census.gov/data/2025/cps/asec/mar/variables.html
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("Childcare source harmonization requires a US Frame.")
    person = frame.table("person").copy()
    required = (
        "person_household_id",
        "person_source_id",
        "age",
        "A_LINENO",
        "PEPAR1",
        "PEPAR2",
        "hours_worked_last_week",
        "PTOTVAL",
        "source_year",
    )
    missing = sorted(set(required) - set(person))
    if missing:
        raise ValueError(f"ASEC childcare source fields are missing: {missing}.")
    if person.duplicated(["person_household_id", "A_LINENO"]).any():
        raise ValueError("ASEC person line numbers must be unique within households.")
    values = person[
        ["age", "A_LINENO", "PEPAR1", "PEPAR2", "hours_worked_last_week"]
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, [0, 4]] < 0).any():
        raise ValueError(
            "ASEC childcare predictors must be finite with nonnegative age/hours."
        )
    if (
        (values[:, :4] % 1 != 0).any()
        or (values[:, 1] <= 0).any()
        or (values[:, 2:4] < -1).any()
    ):
        raise ValueError("ASEC ages and parent/person line numbers are invalid.")
    young = person.age.between(0, 12)
    refs = person.loc[young, ["person_household_id", "PEPAR1", "PEPAR2"]].melt(
        id_vars="person_household_id", value_name="A_LINENO"
    )[["person_household_id", "A_LINENO"]]
    refs = refs.loc[refs.A_LINENO > 0].drop_duplicates()
    parents = refs.merge(
        person[["person_household_id", "A_LINENO", "hours_worked_last_week"]],
        on=["person_household_id", "A_LINENO"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if parents._merge.ne("both").any():
        raise ValueError(
            "A positive ASEC parent pointer does not resolve within its household."
        )
    parents["worked"] = parents.hours_worked_last_week > 0
    counts = parents.groupby("person_household_id").worked.agg(["size", "sum"])
    number = person.person_household_id.map(counts["size"]).fillna(0).to_numpy()
    worked = person.person_household_id.map(counts["sum"]).fillna(0).to_numpy()
    status = np.select(
        [number == 0, worked == 0, worked == number], [-1, 0, 2], default=1
    )
    household = frame.table("household")
    if "household_source_id" in household:
        identities = household.set_index("household_id").household_source_id
        person["childcare_source_household_id"] = person.person_household_id.map(
            identities
        ).astype(str)
    contract = childcare_attendance_contract()
    prices = {
        int(year): value for year, value in contract["cpi_u_annual_average"].items()
    }
    source_prices = person.source_year.map(prices)
    income_values, income_receipts = _income_source_values(person, source_cache)
    if source_prices.isna().any() or not np.isfinite(income_values).all():
        raise ValueError(
            "ASEC childcare income needs finite PTOTVAL; supply the pinned ASEC source cache for omitted raw fields."
        )
    real_income = (
        income_values * prices[contract["income_reference_year"]] / source_prices
    )
    income = real_income.groupby(person.person_household_id).sum()
    income_band = childcare_income_band(person.person_household_id.map(income))
    states = household.set_index("household_id").state_fips
    region = person.person_household_id.map(states).map(
        US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION
    )
    if region.isna().any():
        raise ValueError(
            "Childcare target contains an unknown household/state/region link."
        )
    for name, data in (
        ("parent_work_status", status),
        ("region", region.to_numpy()),
        ("income_band", income_band),
    ):
        if name in person and not np.array_equal(person[name].to_numpy(), data):
            raise ValueError(
                f"Existing childcare predictor {name} disagrees with source harmonization."
            )
        person[name] = data
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            "childcare_predictor_harmonization": {
                "source": "ASEC resident parent line pointers",
                "parent_universe": "parents of any child age 0 through 12 in household",
                "work_measure": "hours_worked_last_week > 0",
                "geography": "Census region from target state_fips",
                "income_source_receipts": income_receipts,
                "income": "Household sum of measured ASEC PTOTVAL, CPI-U adjusted from source year to 2023 dollars",
            },
        },
    )


def _income_source_values(person, source_cache):
    values = person.PTOTVAL.copy()
    receipts = []
    if source_cache is None:
        return values, receipts
    for year in sorted(person.source_year.unique()):
        if year not in ASEC_EDUCATION_ASSISTANCE_ARCHIVES:
            raise ValueError("No pinned ASEC source covers this income year.")
        pin = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year]
        path = Path(source_cache) / pin.member
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != pin.member_sha256 or path.stat().st_size != pin.member_size_bytes:
            raise ValueError("ASEC childcare income source identity mismatch.")
        raw = pd.read_csv(
            path,
            usecols=["PERIDNUM", "PTOTVAL", "A_LINENO", "A_AGE"],
            dtype={"PERIDNUM": str},
        )
        if len(raw) != pin.rows or raw.PERIDNUM.duplicated().any():
            raise ValueError(
                "ASEC childcare income source person keys/count are invalid."
            )
        selected = person.source_year.eq(year)
        keys = person.loc[selected, "PERIDNUM"]
        if not keys.astype(str).str.fullmatch(r"[0-9]{22}").all():
            raise ValueError(
                "ASEC childcare income join requires exact 22-digit PERIDNUM."
            )
        joined = raw.set_index("PERIDNUM").reindex(keys)
        if (
            joined.isna().any().any()
            or not np.array_equal(
                joined.A_LINENO.to_numpy(), person.loc[selected, "A_LINENO"].to_numpy()
            )
            or not np.array_equal(
                joined.A_AGE.to_numpy(), person.loc[selected, "A_AGE"].to_numpy()
            )
        ):
            raise ValueError(
                "ASEC childcare income join fails person/line/age reconciliation."
            )
        observed = values.loc[selected].notna().to_numpy()
        if not np.array_equal(
            values.loc[selected].to_numpy()[observed],
            joined.PTOTVAL.to_numpy()[observed],
        ):
            raise ValueError("Observed ASEC PTOTVAL disagrees with pinned source.")
        values.loc[selected] = joined.PTOTVAL.to_numpy()
        receipts.append(
            {
                "income_year": int(year),
                "sha256": digest,
                "matched_people": int(selected.sum()),
                "source_rows": len(raw),
            }
        )
    return values, receipts
