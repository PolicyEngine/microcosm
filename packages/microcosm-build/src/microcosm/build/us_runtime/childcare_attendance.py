"""Joint, weighted child-care attendance transfer from normalized child donors.

This is an opt-in preparation primitive, not an enabled US build stage. Source
adapters must supply validated child records, survey weights and shared matching
fields. See docs/us-childcare-attendance.md for the source/activation gates.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.frame import Weights

US_CHILDCARE_ATTENDANCE_COLUMNS = (
    "childcare_attending_days_per_month",
    "childcare_days_per_week",
    "childcare_hours_per_day",
)


def childcare_attendance_contract() -> dict:
    return json.loads(
        files("microcosm.build.us")
        .joinpath("childcare_attendance_source.json")
        .read_text()
    )


def childcare_income_band(income):
    """Band household income expressed in the contract's reference-year dollars."""
    values = np.asarray(income, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Childcare income must be observed and finite.")
    return np.searchsorted(
        childcare_attendance_contract()["income_band_upper_bounds"], values, side="left"
    )


def _validate_attendance(table: pd.DataFrame, *, complete: bool) -> None:
    values = table[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].to_numpy(
        dtype=float, na_value=np.nan
    )
    if complete and np.isnan(values).any():
        raise ValueError("Childcare donor attendance must be complete.")
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Childcare attendance must be finite and nonnegative.")
    if (values > np.array([31, 7, 24])).any():
        raise ValueError("Childcare attendance exceeds calendar bounds.")
    monthly = values[:, 0]
    if (monthly[np.isfinite(monthly)] % 1 != 0).any():
        raise ValueError("Childcare attending days per month must be integral.")
    if ((values == 0).any(axis=1) & (values > 0).any(axis=1)).any():
        raise ValueError("Childcare attendance mixes nonattendance and positive care.")


def _ids(table: pd.DataFrame, column: str, *, unique: bool) -> pd.Series:
    ids = table[column]
    if (
        ids.isna().any()
        or not ids.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        raise ValueError(f"{column} must contain nonempty canonical string IDs.")
    if unique and ids.duplicated().any():
        raise ValueError(f"{column} must be unique; donors cannot contain clones.")
    return ids


def impute_us_childcare_attendance(
    person: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    donor_weights: Weights,
    match_columns: tuple[str, ...],
    seed: int,
    fallback_match_columns: tuple[tuple[str, ...], ...] = (),
    sibling_dependence: float = 0.0,
) -> pd.DataFrame:
    """Fill missing attendance for ages 0–12 by a joint weighted donor draw.

    ``donor_weights`` aligns positionally to the donor table. Exact matching must
    include age; other fields (e.g. parental work and region) are supplied by a
    source adapter. No implicit relaxation of an unsupported cell is allowed.
    Nonparticipants belong in the donor pool with three observed zeros. Costs,
    employment, CCDF eligibility and receipt are never used as participation
    flags here.

    Draw once per canonical ``person_source_id``. Merge compatible observations
    across support clones before matching, retaining every observed cell. The
    selected donor supplies all remaining cells jointly, including nonattendance.
    Donor row order, recipient row order and chunk boundaries do not affect the
    draw (provided each source person's clones stay together).

    Return a copy with per-variable ``<variable>_source`` provenance. Missing
    cells outside the source age domain remain null, not inferred zeros. Calling
    this primitive on its own output preserves that provenance and is idempotent.
    """
    if (
        not match_columns
        or "age" not in match_columns
        or len(set(match_columns)) != len(match_columns)
        or set(match_columns) & set(US_CHILDCARE_ATTENDANCE_COLUMNS)
    ):
        raise ValueError("Unique matching columns must include age, not attendance.")
    if not np.isfinite(sibling_dependence) or not 0 <= sibling_dependence <= 1:
        raise ValueError("Sibling dependence must be between zero and one.")
    if sibling_dependence > 0:
        _ids(person, "childcare_source_household_id", unique=False)
    levels = (match_columns, *fallback_match_columns)
    if any(
        "age" not in level
        or len(set(level)) != len(level)
        or not set(level).issubset(match_columns)
        for level in levels
    ):
        raise ValueError(
            "Childcare fallback fields must be unique subsets retaining age."
        )
    for table, required in (
        (person, ("person_source_id", *match_columns)),
        (donor, ("donor_id", *match_columns, *US_CHILDCARE_ATTENDANCE_COLUMNS)),
    ):
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(f"Childcare attendance requires columns: {missing}.")
        if not table.columns.is_unique:
            raise ValueError("Childcare attendance requires unique column names.")
    _ids(person, "person_source_id", unique=False)
    _ids(donor, "donor_id", unique=True)
    if not isinstance(donor_weights, Weights):
        raise TypeError("Childcare donors require typed survey Weights.")
    if len(donor_weights) != len(donor):
        raise ValueError("Childcare donor weights must align with donor rows.")

    result = person.copy(deep=True).reset_index(drop=True)
    pool = donor.copy(deep=True).reset_index(drop=True)
    for table in (result, pool):
        age = pd.to_numeric(table["age"], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
        if not np.isfinite(age).all() or (age < 0).any() or (age % 1 != 0).any():
            raise ValueError("Childcare attendance requires finite whole-year ages.")
        table["age"] = age
    if (pool["age"] > 12).any():
        raise ValueError("Childcare donors must be children aged 0–12.")
    children = result["age"] <= 12
    if (
        pool[list(match_columns)].isna().any().any()
        or result.loc[children, list(match_columns)].isna().any().any()
    ):
        raise ValueError("Childcare matching fields must be complete for children.")
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        if column not in result:
            result[column] = np.nan
        for table in (result, pool):
            table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
        provenance = f"{column}_source"
        if provenance not in result:
            result[provenance] = pd.Series(pd.NA, index=result.index, dtype="string")
        observed = result[column].notna() & result[provenance].isna()
        result.loc[observed, provenance] = "observed"
    _validate_attendance(result, complete=False)
    _validate_attendance(pool, complete=True)
    pool["_donor_weight"] = donor_weights.values
    order = (
        [
            US_CHILDCARE_ATTENDANCE_COLUMNS[1],
            US_CHILDCARE_ATTENDANCE_COLUMNS[2],
            "donor_id",
        ]
        if sibling_dependence > 0
        else ["donor_id"]
    )
    pool = pool.sort_values(order).reset_index(drop=True)
    groups = [
        pool.groupby(list(level), sort=False, dropna=False).indices for level in levels
    ]
    if fallback_match_columns and "childcare_attendance_match_level" not in result:
        result["childcare_attendance_match_level"] = pd.Series(
            pd.NA, index=result.index, dtype="string"
        )

    for source_id, rows in result.groupby(
        "person_source_id", sort=False
    ).groups.items():
        replicas = result.loc[rows]
        if replicas[list(match_columns)].drop_duplicates().shape[0] != 1:
            raise ValueError(f"Childcare clone matching fields disagree: {source_id}.")
        if replicas["age"].iloc[0] > 12:
            continue
        known: dict[str, float] = {}
        for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
            observed = replicas[column].dropna().unique()
            if len(observed) > 1:
                raise ValueError(f"Childcare clone observations disagree: {source_id}.")
            if len(observed):
                known[column] = float(observed[0])
        _validate_attendance(
            pd.DataFrame([known], columns=US_CHILDCARE_ATTENDANCE_COLUMNS),
            complete=False,
        )
        missing = replicas[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].isna()
        if not missing.any().any():
            continue
        if len(known) == len(US_CHILDCARE_ATTENDANCE_COLUMNS):
            selected = known
            source = f"source_person:{source_id}"
        else:
            for level, group in zip(levels, groups, strict=True):
                key = tuple(replicas[column].iloc[0] for column in level)
                group_key = key[0] if len(key) == 1 else key
                candidates = pool.iloc[group.get(group_key, [])]
                candidates = candidates.loc[candidates["_donor_weight"] > 0]
                for column, value in known.items():
                    candidates = candidates.loc[candidates[column] == value]
                if not candidates.empty:
                    break
            if candidates.empty:
                raise ValueError(
                    f"No compatible positive-weight childcare donor: {source_id}."
                )
            # Scale before summing to avoid overflow from large survey weights.
            weights = candidates["_donor_weight"].to_numpy(dtype=float)
            weights = weights / weights.max()
            cumulative = np.cumsum(weights / weights.sum())
            payload = json.dumps([int(seed), "childcare_attendance", source_id])
            if sibling_dependence > 0:
                household_ids = replicas.childcare_source_household_id.unique()
                if len(household_ids) != 1:
                    raise ValueError(
                        "Childcare source clones disagree about household identity."
                    )
                household = household_ids[0]
                selection = hashlib.sha256(
                    json.dumps([int(seed), "sibling_mixture", household]).encode()
                ).digest()
                if (
                    int.from_bytes(selection[:8], "big") >> 11
                ) / 2**53 < sibling_dependence:
                    payload = json.dumps(
                        [int(seed), "childcare_household_rank", household]
                    )
            digest = hashlib.sha256(payload.encode()).digest()
            draw = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
            position = min(
                int(np.searchsorted(cumulative, draw, side="right")),
                len(candidates) - 1,
            )
            selected = candidates.iloc[position]
            source = f"donor:{selected['donor_id']}"
            if fallback_match_columns:
                result.loc[rows, "childcare_attendance_match_level"] = ",".join(level)
        for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
            empty_rows = replicas.index[replicas[column].isna()]
            result.loc[empty_rows, column] = selected[column]
            result.loc[empty_rows, f"{column}_source"] = source

    _validate_attendance(result, complete=False)
    result[US_CHILDCARE_ATTENDANCE_COLUMNS[0]] = result[
        US_CHILDCARE_ATTENDANCE_COLUMNS[0]
    ].astype("Int64")
    # Preserve caller indices, ordering and all unrelated source columns.
    result.index = person.index
    result["age"] = person["age"]
    return result
