"""UK local-area target surface and household metric contract.

The functions here define the household-side metrics that Microcosm local UK
builds calibrate against for Westminster constituencies and local authorities.
They intentionally accept a PolicyEngine-UK-like simulation object rather than
importing a country data package. Target values are supplied separately as
area-indexed tables and consumed by :mod:`microcosm.build.uk_runtime.local_geography`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import resources as importlib_resources
from typing import Any

import numpy as np
import pandas as pd

AREA_TYPES = ("constituency", "la")
AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL = {
    "constituency": "constituency",
    "la": "local_authority",
}
INCOME_VARIABLES = ("self_employment_income", "employment_income")
AGE_BANDS = tuple((lower, lower + 10) for lower in range(0, 80, 10))
CONSTITUENCY_UC_CHILDREN_METRICS = (
    "uc_hh_0_children",
    "uc_hh_1_child",
    "uc_hh_2_children",
    "uc_hh_3plus_children",
)
LA_EXTRA_METRICS = (
    "ons/equiv_net_income_bhc",
    "ons/equiv_net_income_ahc",
    "ons/equiv_housing_costs",
    "tenure/owned_outright",
    "tenure/owned_mortgage",
    "tenure/private_rent",
    "tenure/social_rent",
    "rent/private_rent",
)
COUNCIL_TAX_BANDS = tuple("ABCDEFGH")
COUNTRY_TO_REGION = {
    "England": "SOUTH_EAST",
    "Scotland": "SCOTLAND",
    "Wales": "WALES",
    "Northern Ireland": "NORTHERN_IRELAND",
}
UK_POPULATION_TARGETS_RESOURCE = "uk_population_targets.json"


def load_uk_population_contract() -> dict[str, Any]:
    """Load the packaged UK population Ledger target contract."""

    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_POPULATION_TARGETS_RESOURCE)
        .read_text()
    )
    return json.loads(payload)


def load_uk_local_geography_contract() -> dict[str, Any]:
    """Load the full UK population contract used by local-geography filters."""

    return load_uk_population_contract()


def metric_names(
    area_type: str,
    *,
    target_profile: Mapping[str, Any] | Any | None = None,
) -> tuple[str, ...]:
    """Ordered local-area calibration metric names for ``area_type``."""

    _validate_area_type(area_type)
    if target_profile is not None:
        return metric_names_from_target_profile(target_profile, area_type)
    names: list[str] = []
    for income_variable in INCOME_VARIABLES:
        names.append(f"hmrc/{income_variable}/amount")
        names.append(f"hmrc/{income_variable}/count")
    names.extend(f"age/{lower}_{upper}" for lower, upper in AGE_BANDS)
    names.append("uc_households")
    if area_type == "constituency":
        names.extend(CONSTITUENCY_UC_CHILDREN_METRICS)
    else:
        names.extend(LA_EXTRA_METRICS)
    # This list is append-only: adding a metric must never renumber the
    # metric indices already in the surface, because consumers address them
    # positionally. "households" was appended under that rule and keeps its
    # index; later families land after it rather than displacing it.
    names.append("households")
    if area_type == "la":
        names.extend(f"council_tax/band_{band.lower()}" for band in COUNCIL_TAX_BANDS)
    return tuple(names)


def metric_names_from_target_profile(
    target_profile: Mapping[str, Any] | Any,
    area_type: str,
    *,
    backend: str = "policyengine",
) -> tuple[str, ...]:
    """Ordered UK local metric names from a Ledger target profile.

    The profile is duck-typed so Microcosm can consume a Ledger-exported JSON
    object or the Ledger package's parsed dataclasses without importing Ledger
    at runtime. Microcosm still owns calculation of the backend binding; Ledger
    owns which target contracts are active for each geography level.
    """

    _validate_area_type(area_type)
    geography_level = AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL[area_type]
    names: list[str] = []
    for target in _profile_targets(target_profile):
        geography_levels = tuple(_profile_get(target, "geography_levels", ()))
        if geography_level not in geography_levels:
            continue
        binding = _profile_binding(target, backend)
        metric_name = _profile_get(binding, "metric_name", "")
        if not isinstance(metric_name, str) or not metric_name:
            target_id = _profile_get(target, "target_id", "<unknown>")
            raise ValueError(
                f"Ledger target profile row {target_id!r} has no non-empty "
                f"{backend!r} metric_name binding."
            )
        names.append(metric_name)

    if not names:
        raise ValueError(
            f"Ledger target profile has no {backend!r} metrics for "
            f"area_type {area_type!r}."
        )
    duplicate_names = sorted(
        name for name in set(names) if sum(candidate == name for candidate in names) > 1
    )
    if duplicate_names:
        raise ValueError(
            "Ledger target profile declares duplicate metric binding(s): "
            f"{duplicate_names}."
        )
    return tuple(names)


def area_groups_from_codes(
    area_codes: pd.DataFrame | Sequence[str],
    *,
    code_column: str = "code",
    group_column: str = "country",
) -> dict[str, str]:
    """Map area code to country/devolution group.

    If a ``country`` column is available it is authoritative; otherwise the ONS
    area-code prefix is used.
    """

    if isinstance(area_codes, pd.DataFrame):
        codes = [_normalize_area_code(code) for code in area_codes[code_column]]
        if group_column in area_codes.columns:
            groups = area_codes[group_column].astype(str).tolist()
        else:
            groups = [_country_from_area_code(code) for code in codes]
    else:
        codes = [_normalize_area_code(code) for code in area_codes]
        groups = [_country_from_area_code(code) for code in codes]
    return dict(zip(codes, groups, strict=True))


def metric_tables_by_area_group(
    sim_by_group: Mapping[str, Any],
    area_type: str,
    *,
    period: int | str | None = None,
    household_ids: Sequence[Any] | None = None,
    target_profile: Mapping[str, Any] | Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute metric tables for each country/devolution group simulation."""

    return {
        group: compute_household_metrics(
            sim,
            area_type,
            period=period,
            household_ids=household_ids,
            target_profile=target_profile,
        )
        for group, sim in sim_by_group.items()
    }


def compute_household_metrics(
    sim: Any,
    area_type: str,
    *,
    period: int | str | None = None,
    household_ids: Sequence[Any] | None = None,
    target_profile: Mapping[str, Any] | Any | None = None,
) -> pd.DataFrame:
    """Compute local calibration metrics on household rows.

    ``sim`` must provide ``calculate(variable, period=...)`` and
    ``map_result(values, from_entity, to_entity)`` methods compatible with
    PolicyEngine-UK's microsimulation API. UC child-band membership is
    evaluated for each benefit unit and summed onto its household row, so a
    multi-benefit-unit household can contribute to more than one band.
    """

    _validate_area_type(area_type)
    matrix = pd.DataFrame()

    for income_variable in INCOME_VARIABLES:
        income_values = _values(_calculate(sim, income_variable, period))
        in_spi_frame = _values(_calculate(sim, "income_tax", period)) > 0
        matrix[f"hmrc/{income_variable}/amount"] = _map_result(
            sim,
            income_values * in_spi_frame,
            "person",
            "household",
        )
        matrix[f"hmrc/{income_variable}/count"] = _map_result(
            sim,
            (income_values != 0) * in_spi_frame,
            "person",
            "household",
        )

    age = _values(_calculate(sim, "age", period))
    for lower, upper in AGE_BANDS:
        in_band = (age >= lower) & (age < upper)
        matrix[f"age/{lower}_{upper}"] = _map_result(
            sim,
            in_band,
            "person",
            "household",
        )

    matrix["households"] = np.ones(len(matrix), dtype=float)

    on_uc = (_values(_calculate(sim, "universal_credit", period)) > 0).astype(float)
    on_uc_hh = _map_result(sim, on_uc, "benunit", "household")
    matrix["uc_households"] = on_uc_hh

    if area_type == "constituency":
        num_children = _values(_calculate(sim, "num_children", period))
        on_uc_bool = on_uc > 0
        child_bands = (
            ("uc_hh_0_children", num_children == 0),
            ("uc_hh_1_child", num_children == 1),
            ("uc_hh_2_children", num_children == 2),
            ("uc_hh_3plus_children", num_children >= 3),
        )
        for metric, in_band in child_bands:
            matrix[metric] = _map_result(
                sim,
                on_uc_bool & in_band,
                "benunit",
                "household",
            )

    if area_type == "la":
        hbai_net_income = _values(
            _calculate(sim, "equiv_hbai_household_net_income", period)
        )
        hbai_net_income_ahc = _values(
            _calculate(sim, "equiv_hbai_household_net_income_ahc", period)
        )
        housing_costs = hbai_net_income - hbai_net_income_ahc

        matrix["ons/equiv_net_income_bhc"] = hbai_net_income
        matrix["ons/equiv_net_income_ahc"] = hbai_net_income_ahc
        matrix["ons/equiv_housing_costs"] = housing_costs

        tenure_type = _values(_calculate(sim, "tenure_type", period))
        matrix["tenure/owned_outright"] = (tenure_type == "OWNED_OUTRIGHT").astype(
            float
        )
        matrix["tenure/owned_mortgage"] = (tenure_type == "OWNED_WITH_MORTGAGE").astype(
            float
        )
        matrix["tenure/private_rent"] = (tenure_type == "RENT_PRIVATELY").astype(float)
        matrix["tenure/social_rent"] = (
            (tenure_type == "RENT_FROM_COUNCIL") | (tenure_type == "RENT_FROM_HA")
        ).astype(float)

        is_private_renter = (tenure_type == "RENT_PRIVATELY").astype(float)
        benunit_rent = _values(_calculate(sim, "benunit_rent", period))
        household_rent = _map_result(sim, benunit_rent, "benunit", "household")
        matrix["rent/private_rent"] = household_rent * is_private_renter

        council_tax_band = _values(_calculate(sim, "council_tax_band", period))
        for band in COUNCIL_TAX_BANDS:
            matrix[f"council_tax/band_{band.lower()}"] = (
                council_tax_band == band
            ).astype(float)

    expected = metric_names(area_type, target_profile=target_profile)
    matrix = matrix.loc[:, expected]
    if household_ids is None:
        household_ids = _values(
            _calculate(sim, "household_id", period, map_to="household")
        )
    if len(household_ids) != len(matrix):
        raise ValueError(
            "household_ids must align with household metric rows, got "
            f"{len(household_ids)} ids for {len(matrix)} rows."
        )
    matrix.index = pd.Index(household_ids)
    if matrix.index.has_duplicates:
        duplicates = matrix.index[matrix.index.duplicated()].unique()
        raise ValueError(
            "household metric index must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return matrix


def _profile_targets(target_profile: Mapping[str, Any] | Any) -> tuple[Any, ...]:
    if hasattr(target_profile, "targets"):
        targets = target_profile.targets
    elif isinstance(target_profile, Mapping):
        targets = target_profile.get("targets")
    else:
        targets = None
    if not isinstance(targets, Sequence) or isinstance(targets, str | bytes):
        raise ValueError("Ledger target profile must expose a target sequence.")
    return tuple(targets)


def _profile_binding(target: Any, backend: str) -> Any:
    if hasattr(target, "binding"):
        try:
            return target.binding(backend)
        except KeyError as exc:
            target_id = _profile_get(target, "target_id", "<unknown>")
            raise ValueError(
                f"Ledger target profile row {target_id!r} has no {backend!r} binding."
            ) from exc
    bindings = _profile_get(target, "bindings", {})
    if not isinstance(bindings, Mapping):
        target_id = _profile_get(target, "target_id", "<unknown>")
        raise ValueError(
            f"Ledger target profile row {target_id!r} has invalid bindings."
        )
    try:
        return bindings[backend]
    except KeyError as exc:
        target_id = _profile_get(target, "target_id", "<unknown>")
        raise ValueError(
            f"Ledger target profile row {target_id!r} has no {backend!r} binding."
        ) from exc


def _profile_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _validate_area_type(area_type: str) -> None:
    if area_type not in AREA_TYPES:
        raise ValueError(f"area_type must be one of {AREA_TYPES}, got {area_type!r}.")


def _country_from_area_code(code: str) -> str:
    code = _normalize_area_code(code)
    prefix = code[0].upper()
    by_prefix = {
        "E": "England",
        "W": "Wales",
        "S": "Scotland",
        "N": "Northern Ireland",
    }
    try:
        return by_prefix[prefix]
    except KeyError:
        raise ValueError(
            f"Unknown UK local-area code prefix {prefix!r} for code {code!r}."
        ) from None


def _normalize_area_code(code: Any) -> str:
    if code is None or pd.isna(code):
        raise ValueError("UK local-area code must not be missing.")
    code = str(code).strip()
    if not code:
        raise ValueError("UK local-area code must not be blank.")
    return code


def _calculate(
    sim: Any,
    variable: str,
    period: int | str | None,
    *,
    map_to: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if period is not None:
        kwargs["period"] = period
    if map_to is not None:
        kwargs["map_to"] = map_to
    return sim.calculate(variable, **kwargs)


def _values(result: Any) -> np.ndarray:
    if hasattr(result, "values"):
        return np.asarray(result.values)
    return np.asarray(result)


def _map_result(
    sim: Any,
    values: Any,
    from_entity: str,
    to_entity: str,
) -> np.ndarray:
    return np.asarray(sim.map_result(values, from_entity, to_entity), dtype=float)
