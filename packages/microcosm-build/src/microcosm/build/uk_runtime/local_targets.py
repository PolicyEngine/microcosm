"""UK local-area target surface and household metric contract.

The functions here define the household-side metrics that Microcosm local UK
builds calibrate against for Westminster constituencies and local authorities.
They intentionally accept a PolicyEngine-UK-like simulation object rather than
importing a country data package. Target values are supplied separately as
area-indexed tables and consumed by :mod:`microcosm.build.uk_runtime.local_geography`.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping, Sequence
from importlib import resources as importlib_resources
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.target_materialization import prepared_binding_values
from microcosm.build.uk_runtime.council_tax_constants import UK_COUNCIL_TAX_BANDS
from microcosm.build.uk_runtime.high_value_homes import uk_high_value_home_metric_names

#: ``brma`` is Scotland's 18 Broad Rental Market Areas, the grain ONS's Price
#: Index of Private Rents publishes Scottish rents at (microcosm#1090, #1262).
#: It is a partial grain: households outside Scotland carry no BRMA area.
AREA_TYPES = ("constituency", "la", "brma")
AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL = {
    "constituency": "constituency",
    "la": "local_authority",
    "brma": "statistical_scope",
}
#: Area types every household belongs to; a partial grain covers some only.
FULL_AREA_TYPES = ("constituency", "la")
PARTIAL_AREA_TYPES = ("brma",)
#: The BRMA grain's metrics: Scotland's PIPR rents, 12 x the BRMA mean x its
#: private renters, against the same household rent the local-authority rows
#: read.
BRMA_METRICS = ("rent/private_rent_brma",)
INCOME_VARIABLES = ("self_employment_income", "employment_income")
#: SPI higher rate taxpayers by area (microcosm#1123): the engine's tax_band
#: on adjusted net income less allowances, as HMRC classes a taxpayer by the
#: highest rate paid on any income.
HIGHER_RATE_TAXPAYERS_METRIC = "hmrc/higher_rate_taxpayers/count"
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
#: DESNZ sub-national domestic energy by local authority (microcosm#1207):
#: electricity and gas kWh, and the gas-connected households. DESNZ publishes
#: no constituency cut, so the family is local-authority only.
LA_ENERGY_METRICS = (
    "energy/electricity_kwh",
    "energy/gas_kwh",
    "energy/gas_connected_households",
)
#: England's dwellings with the single adult council tax discount by billing
#: authority (MHCLG Council Taxbase line 8; microcosm#1095), the cells of the
#: national single adult row. The return is England's and by billing
#: authority, so the family is local-authority only.
LA_COUNCIL_TAX_DISCOUNT_METRICS = ("council_tax/single_adult_discount",)
COUNTRY_TO_REGION = {
    "England": "SOUTH_EAST",
    "Scotland": "SCOTLAND",
    "Wales": "WALES",
    "Northern Ireland": "NORTHERN_IRELAND",
}
UK_POPULATION_TARGETS_RESOURCE = "uk_population_targets.json"
UK_LOCAL_AREA_CROSSWALK_RESOURCE = "local_area_crosswalk.json"


def load_uk_population_contract() -> dict[str, Any]:
    """Load the packaged UK population Ledger target contract."""

    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_POPULATION_TARGETS_RESOURCE)
        .read_text()
    )
    return json.loads(payload)


def _brma_name_by_area() -> dict[str, str]:
    crosswalk = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_LOCAL_AREA_CROSSWALK_RESOURCE)
        .read_text()
    )
    level = crosswalk["levels"][AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL["brma"]]
    return {str(code): str(name) for code, name in level["brma_name_by_area"].items()}


def uk_brma_area_roster() -> tuple[str, ...]:
    """The BRMA grain's area codes, Scotland's 18 PIPR BRMAs (``S33...``)."""

    return tuple(sorted(_brma_name_by_area()))


def uk_brma_area_codes(brma: Sequence[object]) -> np.ndarray:
    """Each household's BRMA area code on the BRMA grain, blank outside it.

    The crosswalk names the BRMA of each of the grain's codes
    (``brma_name_by_area``, from the area-to-BRMA layer), and a placed
    household reads its BRMA off its assigned area (microcosm#1262), so a
    Scottish household's code is its BRMA's; any other household belongs to
    no area of the grain.
    """

    code_by_name = {name: code for code, name in _brma_name_by_area().items()}
    return np.asarray(
        [code_by_name.get(str(value), "") for value in brma], dtype=object
    )


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
    if area_type == "brma":
        return BRMA_METRICS
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
    # positionally. Chronicle-backed "households" was appended under that
    # rule and keeps its index; later families land after it rather than
    # displacing it.
    names.append("households")
    if area_type == "la":
        names.extend(
            f"council_tax/band_{band.lower()}" for band in UK_COUNCIL_TAX_BANDS
        )
    names.append(HIGHER_RATE_TAXPAYERS_METRIC)
    # microcosm#1240: owner-occupied homes above value thresholds, both grains,
    # appended after every earlier family under the same append-only rule.
    names.extend(uk_high_value_home_metric_names())
    # microcosm#1207: DESNZ sub-national domestic energy, local authorities
    # only, appended after every earlier family under the same rule.
    if area_type == "la":
        names.extend(LA_ENERGY_METRICS)
    # microcosm#1095: England's single adult council tax discount by billing
    # authority, local authorities only, appended under the same rule.
    if area_type == "la":
        names.extend(LA_COUNCIL_TAX_DISCOUNT_METRICS)
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
    scopes_by_name: dict[str, list[frozenset[str] | None]] = {}
    duplicate_names: set[str] = set()
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
        scope = _profile_area_scope_prefixes(target, geography_level)
        if metric_name not in scopes_by_name:
            scopes_by_name[metric_name] = [scope]
            names.append(metric_name)
            continue
        # One metric bound by several nation-scoped families is one column
        # (microcosm#929: the council-tax bands come from MHCLG, StatsWales
        # and CTAXBASE, each scoped to its nation's GSS prefix); the area's
        # prefix selects the target at bind time. Two rows that could claim
        # the same area are still a duplicate.
        if scope is None or any(
            seen is None or seen & scope for seen in scopes_by_name[metric_name]
        ):
            duplicate_names.add(metric_name)
        scopes_by_name[metric_name].append(scope)

    if not names:
        raise ValueError(
            f"Ledger target profile has no {backend!r} metrics for "
            f"area_type {area_type!r}."
        )
    if duplicate_names:
        raise ValueError(
            "Ledger target profile declares duplicate metric binding(s): "
            f"{sorted(duplicate_names)}."
        )
    return tuple(names)


def _profile_area_scope_prefixes(
    target: Mapping[str, Any] | Any, geography_level: str
) -> frozenset[str] | None:
    """The GSS prefixes a profile row's ``area_scope`` claims at one level.

    ``None`` when the row declares no scope for the level (it claims every
    area of the level).
    """

    area_scope = _profile_get(target, "area_scope", None)
    if not isinstance(area_scope, Mapping):
        return None
    level_scope = area_scope.get(geography_level)
    if not isinstance(level_scope, Mapping):
        return None
    prefixes = frozenset(str(prefix) for prefix in level_scope.get("gss_prefixes", ()))
    return prefixes or None


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
    resolver_by_group: Mapping[str, Any],
    area_type: str,
    *,
    period: int | str | None = None,
    household_ids: Sequence[Any] | None = None,
    target_profile: Mapping[str, Any] | Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute metric tables for each country/devolution group's resolver."""

    return {
        group: compute_household_metrics(
            resolver,
            area_type,
            period=period,
            household_ids=household_ids,
            target_profile=target_profile,
        )
        for group, resolver in resolver_by_group.items()
    }


#: The binding keys the local line honours. A local declaration that carries
#: any other (a measurement period, a band, a reduction) refuses rather than
#: being read without it.
_LOCAL_BINDING_KEYS = frozenset(
    {
        "metric_name",
        "notes",
        "value_variable",
        "value_expression",
        "from_entity",
        "map_to",
        "filters",
    }
)
#: The filter keys the local line honours: one test on a column of the
#: binding's own entity. A filter that names another entity, a concept or a
#: reduction refuses, since the local line reads it as none of those.
_LOCAL_FILTER_KEYS = frozenset(
    {"variable", "operator", "value", "equals", "in", "lower", "upper"}
)


def local_metric_bindings(area_type: str) -> dict[str, Mapping[str, Any]]:
    """Each local metric's ``policyengine`` binding for ``area_type``.

    From the packaged contract's local declarations. A metric that several
    nation-scoped families declare (the council-tax bands of MHCLG,
    StatsWales and CTAXBASE) must bind the same way in each, and every binding
    sums onto the household with the keys the local line honours.
    """

    _validate_area_type(area_type)
    return dict(_local_metric_bindings(area_type))


# Cached on the packaged contract: a test that patches
# ``load_uk_population_contract`` calls ``_local_metric_bindings.cache_clear()``
# before and after, or it reads the bindings of the contract before it.
@functools.cache
def _local_metric_bindings(area_type: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    level = AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL[area_type]
    bindings: dict[str, Mapping[str, Any]] = {}
    for target in load_uk_population_contract()["targets"]:
        if level not in target.get("geography_levels", ()):
            continue
        binding = target["bindings"]["policyengine"]
        name = str(binding["metric_name"])
        unhonoured = sorted(set(binding) - _LOCAL_BINDING_KEYS)
        if unhonoured or binding.get("map_to", "household") != "household":
            raise ValueError(
                f"local metric {name!r} declares {unhonoured or binding['map_to']!r}, "
                "which the local line does not honour: its rows are households."
            )
        for predicate in binding.get("filters", ()):
            unhonoured = sorted(set(predicate) - _LOCAL_FILTER_KEYS)
            if unhonoured:
                raise ValueError(
                    f"local metric {name!r} has a filter that declares "
                    f"{unhonoured!r}, which the local line does not honour: "
                    "each filter tests a column of the binding's own entity."
                )
        if name in bindings and _binding_terms(bindings[name]) != _binding_terms(
            binding
        ):
            raise ValueError(
                f"local metric {name!r} is declared with two different bindings."
            )
        bindings.setdefault(name, binding)
    return tuple(bindings.items())


def _binding_terms(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in binding.items()
        if key not in ("metric_name", "notes")
    }


class _ResolverColumns:
    """A frame's columns, each read once through the measure resolver."""

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver
        self._rows = {
            entity: len(resolver.frame.table(entity))
            for entity in ("person", "benunit", "household")
        }
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def column(self, entity: str, variable: str) -> np.ndarray:
        # A count is the all-ones indicator over its own entity, as the
        # national adapter reads it.
        if variable == f"{entity}_count":
            return np.ones(self._rows[entity], dtype=float)
        key = (entity, variable)
        if key not in self._cache:
            if not self._resolver.knows(entity, variable):
                raise KeyError(f"{entity}.{variable}")
            values, _route = self._resolver.compute(entity, variable)
            self._cache[key] = np.asarray(values)
        return self._cache[key]


def compute_household_metrics(
    resolver: Any,
    area_type: str,
    *,
    period: int | str | None = None,
    household_ids: Sequence[Any] | None = None,
    target_profile: Mapping[str, Any] | Any | None = None,
    input_period: int | str | None = None,
) -> pd.DataFrame:
    """Compute local calibration metrics on household rows from their declarations.

    Each metric is its declaration's ``policyengine`` binding in
    ``uk_population_targets.json``, read by the national rows' binding
    interpreter (:func:`prepared_binding_values`) on columns the measure
    resolver computes. The local and national lines therefore share one
    interpreter and one resolver, and a rebind moves every line that reads the
    declaration. A binding at the person or benefit-unit grain sums onto its
    household through the simulation's own ``map_result``.

    ``resolver`` is a :class:`UKMeasureResolver` over the frame the rows
    belong to, at the calibration ``period``. The measures that read stored
    inputs at the dataset's period (the high-value homes and the domestic
    energy rows) take that period from the resolver's frame, so
    ``input_period``, when given, must name it.
    """

    _validate_area_type(area_type)
    if period is not None:
        resolver.validate_period(period)
    frame_period = (getattr(resolver.frame, "metadata", None) or {}).get("time_period")
    if (
        input_period is not None
        and frame_period is not None
        and str(input_period) != str(frame_period)
    ):
        raise ValueError(
            f"input_period {input_period!r} is not the frame's period "
            f"{frame_period!r}, which the stored-input measures read."
        )
    names = metric_names(area_type, target_profile=target_profile)
    bindings = local_metric_bindings(area_type)
    undeclared = [name for name in names if name not in bindings]
    if undeclared:
        raise ValueError(f"local metrics without a declaration: {undeclared}.")
    columns = _ResolverColumns(resolver)
    matrix = {}
    for name in names:
        binding = bindings[name]
        entity = str(binding.get("from_entity") or "household")
        values = prepared_binding_values(columns, entity, binding)
        matrix[name] = (
            np.asarray(values, dtype=float)
            if entity == "household"
            else _map_result(resolver.simulation, values, entity, "household")
        )
    return _indexed_metrics(
        pd.DataFrame(matrix), resolver.frame, area_type, household_ids, target_profile
    )


def _indexed_metrics(
    matrix: pd.DataFrame,
    frame: Any,
    area_type: str,
    household_ids: Sequence[Any] | None,
    target_profile: Mapping[str, Any] | Any | None,
) -> pd.DataFrame:
    """Select the area type's metrics in order and index them by household."""

    expected = metric_names(area_type, target_profile=target_profile)
    matrix = matrix.loc[:, expected]
    if household_ids is None:
        household_ids = frame.table("household")["household_id"].to_numpy()
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


def _map_result(
    sim: Any,
    values: Any,
    from_entity: str,
    to_entity: str,
) -> np.ndarray:
    return np.asarray(sim.map_result(values, from_entity, to_entity), dtype=float)
