"""US congressional district geography assignment from Ledger facts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import Frame

SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID = (
    "irs_soi.ty2023.congressional_district_2022.all_returns"
)
CONGRESSIONAL_DISTRICT_GEOID_COLUMN = "congressional_district_geoid"


def congressional_district_distribution_from_ledger_facts(
    facts: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build state-constrained CD sampling weights from SOI CD Ledger facts.

    The SOI congressional-district table has true district rows for multi-
    district states and state-total rows for all states. For at-large states,
    the state-total row is the only usable district proxy; it is encoded using
    the existing Microcosm convention ``state_fips * 100 + 00``.
    """

    district_rows: list[dict[str, Any]] = []
    state_total_rows: list[dict[str, Any]] = []
    for fact in facts:
        if not _is_cd_return_count_fact(fact):
            continue
        value = _numeric_value(fact)
        if value <= 0.0:
            continue
        geography = _mapping_at(fact, "geography")
        geography_id = str(geography.get("id") or "")
        geography_level = str(geography.get("level") or "")
        layout = _mapping_at(fact, "layout")
        if geography_level == "congressional_district":
            geoid = _parse_congressional_district_geoid(geography_id)
            if geoid is None:
                continue
            district_rows.append(
                _distribution_row(
                    state_fips=geoid[:2],
                    congressional_district_geoid=geoid,
                    sampling_weight=value,
                    fact=fact,
                    source_geography_id=geography_id,
                    is_state_total_proxy=False,
                )
            )
        elif geography_level == "state" and str(
            layout.get("groupby_value_id") or ""
        ).endswith("_total"):
            state_fips = _parse_state_fips(geography_id)
            if state_fips is None:
                continue
            state_total_rows.append(
                _distribution_row(
                    state_fips=state_fips,
                    congressional_district_geoid=f"{state_fips}00",
                    sampling_weight=value,
                    fact=fact,
                    source_geography_id=geography_id,
                    is_state_total_proxy=True,
                )
            )

    district_states = {row["state_fips"] for row in district_rows}
    rows = [
        *district_rows,
        *(
            row
            for row in state_total_rows
            if str(row["state_fips"]) not in district_states
        ),
    ]
    if not rows:
        raise ValueError(
            "No SOI congressional-district return-count facts were available."
        )
    distribution = pd.DataFrame(rows).sort_values(
        ["state_fips", CONGRESSIONAL_DISTRICT_GEOID_COLUMN]
    )
    duplicated = distribution[CONGRESSIONAL_DISTRICT_GEOID_COLUMN][
        distribution[CONGRESSIONAL_DISTRICT_GEOID_COLUMN].duplicated()
    ]
    if len(duplicated):
        raise ValueError(
            "Congressional district distribution has duplicate geoid(s): "
            f"{duplicated.head(5).tolist()}."
        )
    return distribution.reset_index(drop=True)


def assign_congressional_districts_to_households(
    household: pd.DataFrame,
    distribution: pd.DataFrame,
    *,
    seed: int = 0,
    state_fips_column: str = "state_fips",
    output_column: str = CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
) -> pd.DataFrame:
    """Assign congressional districts to household rows within each state."""

    if state_fips_column not in household.columns:
        raise ValueError(
            f"household table must contain {state_fips_column!r} before CD assignment."
        )
    prepared_distribution = _prepare_distribution(distribution)
    states = _state_fips_series(
        household[state_fips_column],
        label=state_fips_column,
    ).reset_index(drop=True)
    missing_states = sorted(set(states) - set(prepared_distribution["state_fips"]))
    if missing_states:
        raise ValueError(
            "CD distribution is missing state_fips value(s) present in households: "
            + ", ".join(missing_states[:10])
        )

    assigned = household.copy()
    output = np.empty(len(assigned), dtype=np.int64)
    rng = np.random.default_rng(seed)
    state_groups = states.groupby(states, sort=True).groups
    state_group_order = list(state_groups)
    assert state_group_order == sorted(states.unique().tolist()), (
        "Congressional-district RNG state groups must be sorted unique "
        "normalized state FIPS values."
    )
    for state_fips in state_group_order:
        state_positions = state_groups[state_fips]
        state_distribution = prepared_distribution[
            prepared_distribution["state_fips"] == state_fips
        ]
        weights = state_distribution["sampling_weight"].to_numpy(dtype=np.float64)
        probabilities = weights / weights.sum()
        choices = state_distribution[CONGRESSIONAL_DISTRICT_GEOID_COLUMN].to_numpy(
            dtype=np.int64
        )
        output[np.asarray(state_positions, dtype=np.int64)] = rng.choice(
            choices,
            size=len(state_positions),
            replace=True,
            p=probabilities,
        )
    assigned[output_column] = output
    return assigned


def with_household_congressional_districts(
    frame: Frame,
    distribution: pd.DataFrame,
    *,
    seed: int = 0,
) -> Frame:
    """Return ``frame`` with household congressional districts assigned."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"] = assign_congressional_districts_to_households(
        tables["household"],
        distribution,
        seed=seed,
    )
    for link_name in frame.links:
        tables[link_name] = frame.link(link_name).copy()
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(
        tables,
        frame.schema,
        weights,
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def congressional_district_assignment_summary(
    household: pd.DataFrame,
    distribution: pd.DataFrame,
    *,
    state_fips_column: str = "state_fips",
    cd_column: str = CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
) -> dict[str, Any]:
    """Summarize CD support assignment coverage for provenance logs."""

    assigned_states = _state_fips_series(
        household[state_fips_column],
        label=state_fips_column,
    )
    prepared_distribution = _prepare_distribution(distribution)
    return {
        "applied": cd_column in household.columns,
        "household_rows": int(len(household)),
        "assigned_states": int(assigned_states.nunique()),
        "assigned_congressional_districts": (
            int(pd.Series(household[cd_column]).nunique())
            if cd_column in household.columns
            else 0
        ),
        "distribution_states": int(prepared_distribution["state_fips"].nunique()),
        "distribution_congressional_districts": int(len(prepared_distribution)),
        "state_total_proxy_districts": int(
            prepared_distribution["is_state_total_proxy"].sum()
        ),
    }


def _is_cd_return_count_fact(fact: Mapping[str, Any]) -> bool:
    layout = _mapping_at(fact, "layout")
    observed_measure = _mapping_at(fact, "observed_measure")
    dimensions = _mapping_at(fact, "dimensions")
    return (
        layout.get("record_set_id") == SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID
        and observed_measure.get("source_measure_id") == "return_count"
        and (dimensions.get("filing_status") or "all") == "all"
        and (dimensions.get("income_range") or "all") == "all"
    )


def _distribution_row(
    *,
    state_fips: str,
    congressional_district_geoid: str,
    sampling_weight: float,
    fact: Mapping[str, Any],
    source_geography_id: str,
    is_state_total_proxy: bool,
) -> dict[str, Any]:
    layout = _mapping_at(fact, "layout")
    lineage = _mapping_at(fact, "lineage")
    period = _mapping_at(fact, "period")
    source = _mapping_at(fact, "source")
    return {
        "state_fips": state_fips,
        CONGRESSIONAL_DISTRICT_GEOID_COLUMN: congressional_district_geoid,
        "sampling_weight": float(sampling_weight),
        "source_record_id": str(
            layout.get("source_row_id") or lineage.get("source_record_id") or ""
        ),
        "source_geography_id": source_geography_id,
        "source_sha256": str(source.get("source_sha256") or ""),
        "ledger_period": period.get("value"),
        "is_state_total_proxy": bool(is_state_total_proxy),
    }


def _prepare_distribution(distribution: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state_fips",
        CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
        "sampling_weight",
    }
    missing = sorted(required - set(distribution.columns))
    if missing:
        raise ValueError(f"CD distribution is missing column(s): {missing}.")
    prepared = distribution.copy()
    prepared["state_fips"] = _state_fips_series(
        prepared["state_fips"],
        label="distribution.state_fips",
    )
    prepared[CONGRESSIONAL_DISTRICT_GEOID_COLUMN] = _cd_geoid_series(
        prepared[CONGRESSIONAL_DISTRICT_GEOID_COLUMN]
    )
    geoid_state_fips = prepared[CONGRESSIONAL_DISTRICT_GEOID_COLUMN].map(
        lambda value: f"{int(value) // 100:02d}"
    )
    mismatched = geoid_state_fips != prepared["state_fips"]
    if mismatched.any():
        examples = prepared.loc[
            mismatched,
            ["state_fips", CONGRESSIONAL_DISTRICT_GEOID_COLUMN],
        ].head(5)
        raise ValueError(
            "CD distribution state_fips must match congressional_district_geoid "
            f"prefix; mismatched row(s): {examples.to_dict('records')}."
        )
    weights = pd.to_numeric(prepared["sampling_weight"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("CD distribution sampling_weight must be positive and finite.")
    prepared["sampling_weight"] = weights
    if "is_state_total_proxy" not in prepared.columns:
        prepared["is_state_total_proxy"] = False
    state_weight = prepared.groupby("state_fips")["sampling_weight"].sum()
    invalid_states = state_weight[~np.isfinite(state_weight) | (state_weight <= 0.0)]
    if len(invalid_states):
        raise ValueError(
            "CD distribution has non-positive total sampling weight for state(s): "
            + ", ".join(map(str, invalid_states.index[:5]))
        )
    return prepared


def _state_fips_series(values: Any, *, label: str) -> pd.Series:
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError(f"{label} contains missing values.")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].head(5).tolist()
        raise ValueError(f"{label} contains non-numeric value(s): {bad}.")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    non_finite = ~np.isfinite(numeric_values)
    if non_finite.any():
        bad = series.iloc[np.flatnonzero(non_finite)[:5]].tolist()
        raise ValueError(f"{label} contains non-finite value(s): {bad}.")
    non_integer = numeric_values != np.floor(numeric_values)
    if non_integer.any():
        bad = series.iloc[np.flatnonzero(non_integer)[:5]].tolist()
        raise ValueError(f"{label} must contain integer value(s); invalid: {bad}.")
    integers = pd.Series(numeric_values.astype("int64"), index=series.index)
    invalid = (integers < 1) | (integers > 99)
    if invalid.any():
        raise ValueError(
            f"{label} must contain two-digit state FIPS codes; invalid value(s): "
            f"{integers[invalid].head(5).tolist()}."
        )
    return integers.map(lambda value: f"{int(value):02d}")


def _cd_geoid_series(values: Any) -> pd.Series:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].head(5).tolist()
        raise ValueError(f"CD distribution contains non-numeric geoid(s): {bad}.")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    non_finite = ~np.isfinite(numeric_values)
    if non_finite.any():
        bad = series.iloc[np.flatnonzero(non_finite)[:5]].tolist()
        raise ValueError(f"CD distribution contains non-finite geoid value(s): {bad}.")
    non_integer = numeric_values != np.floor(numeric_values)
    if non_integer.any():
        bad = series.iloc[np.flatnonzero(non_integer)[:5]].tolist()
        raise ValueError(f"CD distribution geoid values must be integers: {bad}.")
    integers = pd.Series(numeric_values.astype("int64"), index=series.index)
    invalid = (integers < 100) | (integers > 9999)
    if invalid.any():
        raise ValueError(
            "CD distribution congressional_district_geoid must be state*100+district; "
            f"invalid value(s): {integers[invalid].head(5).tolist()}."
        )
    return integers


def _parse_congressional_district_geoid(geography_id: str) -> str | None:
    for prefix in ("5001700US", "5001900US"):
        if geography_id.startswith(prefix):
            geoid = geography_id.removeprefix(prefix)
            break
    else:
        return None
    if len(geoid) != 4 or not geoid.isdigit():
        return None
    return geoid


def _parse_state_fips(geography_id: str) -> str | None:
    if not geography_id.startswith("0400000US"):
        return None
    state_fips = geography_id.removeprefix("0400000US")
    if len(state_fips) != 2 or not state_fips.isdigit():
        return None
    return state_fips


def _numeric_value(fact: Mapping[str, Any]) -> float:
    try:
        return float(fact.get("value"))
    except (TypeError, ValueError):
        return 0.0


def _mapping_at(fact: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = fact.get(key)
    if isinstance(value, Mapping):
        return value
    return {}
