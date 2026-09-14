"""UK Ledger target-reference compilation and materialization helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources as importlib_resources
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainRule,
    apply_cross_grain_reconciliation,
)
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    _fact_matches_selector,
    compile_ledger_target_references,
)
from microcosm.build.target_materialization import (
    TargetMaterializationResult,
    materialize_target_bindings,
)
from microcosm.build.uk_runtime.cgt_calibration import uk_cgt_annual_exempt_amount
from microcosm.build.uk_runtime.geography_ladder import UK_ENGLAND_WALES_REGION_CODES
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.local_target_census import family_for_metric
from microcosm.build.uk_runtime.local_targets import (
    AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL,
    area_groups_from_codes,
    load_uk_local_geography_contract,
    metric_names,
)
from microcosm.build.uk_runtime.uc_source_periods import (
    validate_uc_source_month_coverage,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import Frame


@dataclass(frozen=True)
class UKLedgerTargetCompilation:
    """Compiled UK Ledger target registry and unsupported-reference report."""

    registry: TargetRegistry
    unsupported: tuple[dict[str, str], ...]


LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE = "local_registry_parity_fixture_2025.json"
UK_LOCAL_TARGET_REFERENCE_MEMBERSHIP_RESOURCE = "local_target_reference_membership.json"
UK_POPULATION_TARGETS_RESOURCE = "uk_population_targets.json"
UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS = frozenset({"country", "region"})
_UK_LOCAL_FIXTURE_METRIC_ALIASES = {
    f"voa/council_tax/{band}": f"council_tax/band_{band.lower()}" for band in "ABCDEFGH"
}
UK_CENSUS_HOUSEHOLDS_TARGET_ID = "ons.census.households"


def _uk_cross_grain_leg_of_area(area_code: str) -> str:
    # area_groups_from_codes maps code -> country group, so the single value is
    # this code's leg. It refuses an unknown prefix itself; the explicit miss
    # below keeps the refusal fail-closed rather than a bare StopIteration if
    # that mapping ever returns nothing for a code.
    leg = next(iter(area_groups_from_codes((area_code,)).values()), "")
    if not leg:
        raise ValueError(
            f"UK cross-grain area code {area_code!r} maps to no country leg."
        )
    return leg


UK_CROSS_GRAIN_GRAIN_PRECEDENCE = ("country", "constituency", "la")
UK_CROSS_GRAIN_BRIDGES = (
    CrossGrainBridge(
        bridge_id="national_household_composition_partition_vs_census_households",
        concept="uk.household.count",
        higher_target_ids=(
            "ons.household_composition.lone_households_under_65",
            "ons.household_composition.lone_households_over_65",
            "ons.household_composition.unrelated_adult_households",
            "ons.household_composition.couple_no_children_households",
            "ons.household_composition.couple_under_3_children_households",
            "ons.household_composition.couple_3_plus_children_households",
            "ons.household_composition.couple_non_dependent_children_only_households",
            "ons.household_composition.lone_parent_dependent_children_households",
            "ons.household_composition.lone_parent_non_dependent_children_households",
            "ons.household_composition.multi_family_households",
        ),
        lower_side=f"contract:{UK_CENSUS_HOUSEHOLDS_TARGET_ID}",
    ),
    CrossGrainBridge(
        bridge_id="national_uc_caseload_vs_uc_households_by_area",
        concept="uk.benefit_unit.count",
        higher_target_ids=("dwp.uc.households",),
        lower_side="contract:dwp.uc.households_by_area",
    ),
    # The national ONS controls use inclusive integer-age bands (0--9), while
    # local targets use equivalent half-open encodings (0--10), so their
    # signatures cannot match directly. These bridges let the K02000001 UK
    # control rescale both constituency and local-authority bands over its
    # England/Wales/Scotland/Northern Ireland legs.
    CrossGrainBridge(
        bridge_id="national_age_0_9_vs_local_age_0_10",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_0_9_by_region",),
        lower_side="contract:ons.age.0_10",
    ),
    CrossGrainBridge(
        bridge_id="national_age_10_19_vs_local_age_10_20",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_10_19_by_region",),
        lower_side="contract:ons.age.10_20",
    ),
    CrossGrainBridge(
        bridge_id="national_age_20_29_vs_local_age_20_30",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_20_29_by_region",),
        lower_side="contract:ons.age.20_30",
    ),
    CrossGrainBridge(
        bridge_id="national_age_30_39_vs_local_age_30_40",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_30_39_by_region",),
        lower_side="contract:ons.age.30_40",
    ),
    CrossGrainBridge(
        bridge_id="national_age_40_49_vs_local_age_40_50",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_40_49_by_region",),
        lower_side="contract:ons.age.40_50",
    ),
    CrossGrainBridge(
        bridge_id="national_age_50_59_vs_local_age_50_60",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_50_59_by_region",),
        lower_side="contract:ons.age.50_60",
    ),
    CrossGrainBridge(
        bridge_id="national_age_60_69_vs_local_age_60_70",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_60_69_by_region",),
        lower_side="contract:ons.age.60_70",
    ),
    CrossGrainBridge(
        bridge_id="national_age_70_79_vs_local_age_70_80",
        concept="uk.person.count",
        higher_target_ids=("ons.population.age_70_79_by_region",),
        lower_side="contract:ons.age.70_80",
    ),
)
# A future move of these declarations into country-package spec JSON follows
# the country-owned specification direction established in microcosm#159.
UK_CROSS_GRAIN_RULE = CrossGrainRule(
    grain_precedence=UK_CROSS_GRAIN_GRAIN_PRECEDENCE,
    signature_fields=("concept", "entity", "map_to", "filters"),
    bridges=UK_CROSS_GRAIN_BRIDGES,
    leg_of_area=_uk_cross_grain_leg_of_area,
    parent_geography_legs={
        "K02000001": ("England", "Wales", "Scotland", "Northern Ireland"),
        "K03000001": ("England", "Wales", "Scotland"),
        "E92000001": ("England",),
        "W92000004": ("Wales",),
        "S92000003": ("Scotland",),
        "N92000002": ("Northern Ireland",),
    },
)


def align_uk_local_registry_parity_fixture(
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Align incumbent local metric names to the Microcosm contract ids.

    The incumbent fixture is extracted from the old runtime's metric columns
    (``hmrc/self_employment_income/amount@...``). The local Ledger registry is
    contract-named (``hmrc.self_employment_income.amount@...``). In-contract
    fixture rows are therefore renamed for comparison; out-of-contract metrics
    deliberately remain fixture-only so the signed receipt records the ruled
    exclusions.
    """

    metric_target_ids = _uk_local_metric_target_ids()
    rows: list[dict[str, Any]] = []
    for row in fixture.get("rows", ()):
        if not isinstance(row, Mapping):
            rows.append(row)
            continue
        updated = dict(row)
        metric = str(
            updated.get("metric") or str(updated.get("name", "")).split("@")[0]
        )
        contract_metric = _UK_LOCAL_FIXTURE_METRIC_ALIASES.get(metric, metric)
        target_id = metric_target_ids.get(contract_metric)
        if target_id is not None:
            geography_id = str(
                updated.get("geography_id")
                or str(updated.get("name", "")).split("@", 1)[1]
            )
            updated["name"] = f"{target_id}@{geography_id}"
            updated["contract_target_id"] = updated["name"]
            updated.setdefault("measure", metric)
        rows.append(updated)
    aligned = dict(fixture)
    aligned["rows"] = rows
    return aligned


def _uk_local_metric_target_ids() -> dict[str, str]:
    contract = load_uk_local_geography_contract()
    mapping: dict[str, str] = {}
    for target in contract.get("targets", ()):
        if not isinstance(target, Mapping):
            continue
        bindings = target.get("bindings")
        if not isinstance(bindings, Mapping):
            continue
        policyengine = bindings.get("policyengine")
        if not isinstance(policyengine, Mapping):
            continue
        metric_name = policyengine.get("metric_name")
        target_id = target.get("target_id")
        if not isinstance(metric_name, str) or not isinstance(target_id, str):
            continue
        existing = mapping.get(metric_name)
        if existing is not None and existing != target_id:
            raise ValueError(
                "UK local geography contract maps metric "
                f"{metric_name!r} to multiple target ids: "
                f"{existing!r} and {target_id!r}."
            )
        mapping[metric_name] = target_id
    return mapping


def compile_uk_target_registry(
    facts: Iterable[Mapping[str, Any]],
    *,
    target_period: int | str,
) -> UKLedgerTargetCompilation:
    """Compile packaged UK Ledger references against consumer fact rows."""

    fact_rows = tuple(facts)
    spec = load_country_spec("uk")
    _assert_household_type_bindings_declared(_uk_contract_targets())
    compiled = []
    unsupported: list[dict[str, str]] = []
    for reference in spec.target_references:
        restamped = LedgerTargetReference(
            **{**reference.__dict__, "period": target_period}
        )
        candidate_facts = _candidate_facts_for_reference(fact_rows, restamped)
        if restamped.uprating_index is not None and (
            str(restamped.uprating_index) not in UK_UPRATING_APPLIERS
        ):
            # A declared index the runtime cannot apply is a contract error,
            # not a fact gap: refuse the whole compile rather than report it
            # as an unsupported row.
            apply_declared_uk_uprating(
                restamped, registry=TargetRegistry((), country="uk")
            )
        try:
            _assert_national_region_pin(restamped)
            registry = compile_ledger_target_references(
                candidate_facts,
                [restamped],
                country="uk",
            )
            registry = _assert_region_facts_resolved_at_region(restamped, registry)
            registry = apply_declared_uk_uprating(restamped, registry)
            registry = validate_uc_source_month_coverage(
                restamped, registry, candidate_facts
            )
            if reference.name == "hmrc.cgt.liability_total":
                cash_metadata = _cgt_cash_diagnostic_metadata(fact_rows)
                registry = TargetRegistry(
                    (
                        replace(row, metadata={**row.metadata, **cash_metadata})
                        for row in registry.specs
                    ),
                    country="uk",
                )
        except ValueError as error:
            unsupported.append(
                {
                    "name": reference.name,
                    "period": target_period,
                    "reason": str(error),
                }
            )
        else:
            compiled.extend(registry.specs)
    return UKLedgerTargetCompilation(
        TargetRegistry(compiled, country="uk"),
        tuple(unsupported),
    )


#: DfT BUS05ai fare receipts are aligned from the publisher's year-ending-March
#: period to the calibration year with the BUS0415 local bus fares index (a
#: ruling of 2026-09-10: calendar-year basis). The contract declares the index
#: on the fare-receipt rows (``uprating_index``); net support declares none and
#: is never aligned. The index series is read from the vendored resource
#: ``dft_bus_value_anchors.json`` (hash-pinned to the same Chronicle feed as
#: the references), so the factor is reproducible without the licensed feed.
UK_DFT_BUS_FARE_RECEIPTS_CONCEPT = "dft.local_bus_passenger_fare_receipts"
UK_DFT_BUS_FARES_INDEX_CONCEPT = "dft.local_bus_fares_index"
UK_DFT_BUS_FARES_INDEX_RESOURCE = "dft_bus_value_anchors.json"
UK_DFT_BUS_FARE_INDEX_BASIS = (
    "mean of the quarter-end BUS0415 index (March, June, September, December) "
    "inside the calibration calendar year over the mean of the four quarter-ends "
    "inside the receipts' April-to-March fiscal year"
)


def _vendored_fares_index_rows() -> list[Mapping[str, Any]]:
    """BUS0415 rows from the vendored resource, refused if it lags the feed pin."""

    return vendored_rows(
        UK_DFT_BUS_FARES_INDEX_RESOURCE,
        concept=UK_DFT_BUS_FARES_INDEX_CONCEPT,
        period_type="month",
    )


def _fares_index_series(
    rows: Iterable[Mapping[str, Any]], geography_id: str
) -> dict[str, tuple[float, str]]:
    """Month -> (index value, source record id) for one vendored BUS0415 series."""

    series: dict[str, tuple[float, str]] = {}
    for row in rows:
        if row.get("concept") != UK_DFT_BUS_FARES_INDEX_CONCEPT:
            continue
        geography = row.get("geography")
        if not isinstance(geography, Mapping) or geography.get("id") != geography_id:
            continue
        period = row.get("period")
        if not isinstance(period, Mapping) or period.get("type") != "month":
            continue
        month = str(period.get("value"))
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        record_id = str(row.get("source_record_id") or "")
        if month in series and series[month][0] != float(value):
            raise ValueError(
                f"BUS0415 series {geography_id!r} carries two values for {month!r}."
            )
        series[month] = (float(value), record_id)
    return series


def _quarter_end_months(start_year: int, start_month: int) -> tuple[str, ...]:
    months = []
    year, month = start_year, start_month
    for _ in range(4):
        # advance to the next quarter-end month (3, 6, 9, 12) on or after
        # (year, month), then step past it
        while month not in (3, 6, 9, 12):
            month += 1
            if month > 12:
                month, year = 1, year + 1
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return tuple(months)


def align_dft_bus_fare_receipts_to_period(
    reference: LedgerTargetReference,
    registry: TargetRegistry,
    *,
    index_rows: Iterable[Mapping[str, Any]] | None = None,
) -> TargetRegistry:
    """Transport BUS05ai fare receipts to the reference period with BUS0415.

    Applies to references that declare ``uprating_index`` equal to the BUS0415
    concept; any other reference passes through untouched. The generator and
    the runtime both call this on the compiled registry, so the committed
    membership, the parity receipts and the run manifest carry one value. The
    factor is declared on the spec's metadata with the index concept, the
    series geography, the months used, their source record ids and the
    pre-alignment value.
    """

    if reference.uprating_index != UK_DFT_BUS_FARES_INDEX_CONCEPT:
        return registry
    if str(reference.ledger_selector.get("source_concept") or "") != (
        UK_DFT_BUS_FARE_RECEIPTS_CONCEPT
    ):
        raise ValueError(
            f"UK reference {reference.name!r} declares the BUS0415 fares index on "
            f"concept {reference.ledger_selector.get('source_concept')!r}; the "
            f"index transports {UK_DFT_BUS_FARE_RECEIPTS_CONCEPT!r} only."
        )
    if reference.period is None or not str(reference.period).isdigit():
        raise ValueError(
            f"UK reference {reference.name!r}: BUS0415 alignment needs a "
            "calendar-year reference period."
        )
    target_period = int(str(reference.period))
    rows = list(index_rows) if index_rows is not None else _vendored_fares_index_rows()
    aligned = []
    for spec in registry.specs:
        geography_id = str(spec.metadata.get("ledger_geography_id") or "")
        fact_period = str(spec.metadata.get("ledger_fact_period") or "")
        if not geography_id or not fact_period.isdigit():
            raise ValueError(
                f"UK target {spec.name!r}: cannot align fare receipts without a "
                "resolved Ledger geography and fiscal start year."
            )
        start_year = int(fact_period)
        from_months = _quarter_end_months(start_year, 4)
        to_months = _quarter_end_months(target_period, 1)
        series = _fares_index_series(rows, geography_id)
        missing = [m for m in (*from_months, *to_months) if m not in series]
        if missing:
            raise ValueError(
                f"UK target {spec.name!r}: BUS0415 series {geography_id!r} lacks "
                f"quarter-end month(s) {missing}; refusing to align fare receipts."
            )
        from_mean = sum(series[m][0] for m in from_months) / len(from_months)
        to_mean = sum(series[m][0] for m in to_months) / len(to_months)
        if from_mean <= 0 or to_mean <= 0:
            raise ValueError(
                f"UK target {spec.name!r}: BUS0415 index means must be positive."
            )
        factor = to_mean / from_mean
        record_ids = ",".join(series[m][1] for m in (*from_months, *to_months))
        metadata = {
            **spec.metadata,
            "uprating_index": UK_DFT_BUS_FARES_INDEX_CONCEPT,
            "uprating_index_basis": UK_DFT_BUS_FARE_INDEX_BASIS,
            "uprating_index_resource": UK_DFT_BUS_FARES_INDEX_RESOURCE,
            "uprating_index_series_geography_id": geography_id,
            "uprating_index_from_months": ",".join(from_months),
            "uprating_index_to_months": ",".join(to_months),
            "uprating_index_from_mean": f"{from_mean:.15g}",
            "uprating_index_to_mean": f"{to_mean:.15g}",
            "uprating_index_source_record_ids": record_ids,
            "uprating_factor": f"{factor:.15g}",
            "ledger_value_before_alignment": f"{spec.value:.15g}",
            "uprating_adjudication": "microcosm#890 (ruling 2026-09-10: calendar-year basis)",
        }
        aligned.append(replace(spec, value=spec.value * factor, metadata=metadata))
    return TargetRegistry(aligned, country="uk")


#: Every ``uprating_index`` a UK reference may declare, with the applier the
#: generator and the runtime share. A declared index outside this table is a
#: contract error, refused before compilation.
UK_UPRATING_APPLIERS: Mapping[str, Any] = {
    UK_DFT_BUS_FARES_INDEX_CONCEPT: align_dft_bus_fare_receipts_to_period,
}


def apply_declared_uk_uprating(
    reference: LedgerTargetReference, registry: TargetRegistry
) -> TargetRegistry:
    """Apply the reference's declared ``uprating_index`` (identity when none)."""

    if reference.uprating_index is None:
        return registry
    applier = UK_UPRATING_APPLIERS.get(str(reference.uprating_index))
    if applier is None:
        raise ValueError(
            f"UK reference {reference.name!r} declares uprating_index "
            f"{reference.uprating_index!r}, which no UK applier implements "
            f"(known: {sorted(UK_UPRATING_APPLIERS)})."
        )
    return applier(reference, registry)


#: National references may pin a region only from the published English
#: region roster; the codes are the ladder's own (geography_ladder), so no
#: second register carries them.
UK_NATIONAL_REGION_ROSTER: frozenset[str] = frozenset(
    code for code in UK_ENGLAND_WALES_REGION_CODES if code.startswith("E12")
)


def _assert_national_region_pin(reference: LedgerTargetReference) -> None:
    """Refuse a region-pinned national reference outside the region roster."""

    selector = reference.ledger_selector
    level = str(selector.get("geography_level") or "")
    if level != "region":
        return
    geography_id = str(selector.get("geography_id") or "")
    if geography_id not in UK_NATIONAL_REGION_ROSTER:
        raise ValueError(
            f"UK national reference {reference.name!r} pins region "
            f"{geography_id!r}, which is not in the English region roster "
            f"{sorted(UK_NATIONAL_REGION_ROSTER)}."
        )


def _assert_region_facts_resolved_at_region(
    reference: LedgerTargetReference, registry: TargetRegistry
) -> TargetRegistry:
    """A region-pinned reference must resolve a region-stamped fact."""

    if str(reference.ledger_selector.get("geography_level") or "") != "region":
        return registry
    for spec in registry.specs:
        level = str(spec.metadata.get("ledger_geography_level") or "")
        geography_id = str(spec.metadata.get("ledger_geography_id") or "")
        if level != "region" or geography_id not in UK_NATIONAL_REGION_ROSTER:
            raise ValueError(
                f"UK national reference {reference.name!r} resolved a fact at "
                f"{level!r} {geography_id!r}; a region pin must resolve a "
                "region-stamped fact inside the roster."
            )
    return registry


def _cgt_cash_diagnostic_metadata(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[str, str]:
    """Retain the original forecast without adding a cash row to fitting.

    The cash period and exact Chronicle fact identity are consumer declarations,
    independent of the calibration period. The normal TargetSpec metadata path
    carries this receipt into both national and local registries. Unavailable
    diagnostic data must not disable independently observed HMRC targets.
    """
    contract = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_POPULATION_TARGETS_RESOURCE)
        .read_text(encoding="utf-8")
    )
    try:
        declaration = contract["diagnostic_references"]["obr.capital_gains_tax"]
        if declaration["attach_to_target"] != "hmrc.cgt.liability_total":
            raise ValueError("unexpected receiving target")
        reference = LedgerTargetReference(**declaration["reference"])
        if not reference.ledger_fact_key:
            raise ValueError("the original forecast must have an exact fact pin")
        if (
            reference.name != "obr.capital_gains_tax"
            or reference.period_match_policy != "exact"
            or reference.assertion_policy != "allow_source_projection"
            or declaration["required_period_type"] != "fiscal_year"
            or declaration["required_assertion"] != "source_projection"
        ):
            raise ValueError("expected an exactly dated OBR cash forecast declaration")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CGT cash diagnostic declaration: {error}") from error

    metadata = {
        "cgt_cash_diagnostic_role": "diagnostic_only_not_in_fit",
        "cgt_cash_reconciliation_status": "unresolved",
    }
    try:
        candidates = tuple(
            fact
            for fact in _candidate_facts_for_reference(facts, reference)
            if fact.get("aggregate_fact_key") == reference.ledger_fact_key
            and fact.get("period", {}).get("type")
            == declaration["required_period_type"]
            and fact.get("assertion") == declaration["required_assertion"]
        )
        (cash,) = compile_ledger_target_references(
            candidates, [reference], country="uk"
        ).specs
        if cash.metadata["ledger_fact_period"] != str(reference.period):
            raise ValueError("forecast period does not match its declaration")
    except ValueError as error:
        return {
            **metadata,
            "cgt_cash_diagnostic_status": "unavailable",
            "cgt_cash_diagnostic_unavailable_reason": str(error),
            "cgt_cash_diagnostic_expected_fact_key": reference.ledger_fact_key,
            "cgt_cash_diagnostic_expected_period": str(reference.period),
            "cgt_cash_diagnostic_expected_period_type": declaration[
                "required_period_type"
            ],
            "cgt_cash_diagnostic_expected_assertion": declaration["required_assertion"],
        }
    return {
        **metadata,
        "cgt_cash_diagnostic_status": "available",
        "cgt_cash_diagnostic_value_gbp": str(cash.value),
        "cgt_cash_diagnostic_period": str(cash.period),
        "cgt_cash_diagnostic_source": cash.source,
        **{f"cgt_cash_diagnostic_{key}": value for key, value in cash.metadata.items()},
    }


def load_uk_local_area_crosswalk() -> dict[str, Any]:
    """The committed local-area crosswalk (roster + vintages per level)."""

    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text(encoding="utf-8")
    )


def load_uk_local_target_reference_membership() -> dict[str, Any]:
    """Load the committed local target membership and signed deferrals."""

    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_LOCAL_TARGET_REFERENCE_MEMBERSHIP_RESOURCE)
        .read_text(encoding="utf-8")
    )


def _uk_licensed_empty_legs_from_membership(
    membership: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    """Derive wholly deferred target legs from committed membership rosters."""

    areas_by_level = membership.get("areas_by_geography_level")
    if not isinstance(areas_by_level, Mapping):
        raise ValueError(
            "UK local target membership must expose areas_by_geography_level."
        )
    roster_by_level_leg: dict[tuple[str, str], set[str]] = {}
    for geography_level, raw_area_ids in areas_by_level.items():
        if not isinstance(raw_area_ids, (list, tuple)):
            raise ValueError(
                f"UK local target membership roster {geography_level!r} must be a list."
            )
        for raw_area_id in raw_area_ids:
            area_id = str(raw_area_id).strip()
            if not area_id:
                raise ValueError(
                    "UK local target membership rosters must not contain blank "
                    "area ids."
                )
            leg = _uk_cross_grain_leg_of_area(area_id)
            roster_by_level_leg.setdefault((str(geography_level), leg), set()).add(
                area_id
            )

    signed_deferrals = membership.get("signed_deferrals", ())
    if not isinstance(signed_deferrals, (list, tuple)):
        raise ValueError("UK local target membership signed_deferrals must be a list.")
    deferred_by_target_level_leg: dict[tuple[str, str, str], set[str]] = {}
    for deferral in signed_deferrals:
        if not isinstance(deferral, Mapping):
            raise ValueError(
                "UK local target membership signed deferrals must be mappings."
            )
        target_id = str(deferral.get("target_id", "")).strip()
        geography_level = str(deferral.get("geography_level", "")).strip()
        raw_area_ids = deferral.get("area_ids")
        if (
            not target_id
            or not geography_level
            or not isinstance(raw_area_ids, (list, tuple))
        ):
            raise ValueError(
                "UK local target membership signed deferrals must name a "
                "target_id, geography_level, and area_ids list."
            )
        for raw_area_id in raw_area_ids:
            area_id = str(raw_area_id).strip()
            leg = _uk_cross_grain_leg_of_area(area_id)
            roster = roster_by_level_leg.get((geography_level, leg), set())
            if area_id not in roster:
                raise ValueError(
                    "UK local target membership signed deferral area "
                    f"{area_id!r} is absent from the {geography_level!r} roster."
                )
            deferred_by_target_level_leg.setdefault(
                (target_id, geography_level, leg), set()
            ).add(area_id)

    licensed: dict[str, set[str]] = {}
    for (
        target_id,
        geography_level,
        leg,
    ), deferred in deferred_by_target_level_leg.items():
        roster = roster_by_level_leg[(geography_level, leg)]
        if roster and deferred == roster:
            licensed.setdefault(target_id, set()).add(leg)
    return {
        target_id: frozenset(sorted(legs))
        for target_id, legs in sorted(licensed.items())
    }


def compile_uk_local_target_registry(
    facts: Iterable[Mapping[str, Any]],
    *,
    target_period: int | str,
    crosswalk: Mapping[str, Any],
) -> UKLedgerTargetCompilation:
    """Compile packaged UK local-area Ledger references against fact rows."""

    fact_rows = tuple(facts)
    local_fact_buckets = _local_fact_buckets(fact_rows)
    spec = load_country_spec("uk")
    rosters = _local_crosswalk_rosters(crosswalk)
    compiled = []
    unsupported: list[dict[str, str]] = []
    for reference in spec.local_target_references:
        restamped = LedgerTargetReference(
            **{**reference.__dict__, "period": target_period}
        )
        _assert_local_reference_in_crosswalk(restamped, rosters)
        candidate_facts = _candidate_facts_for_reference(
            _local_candidate_fact_pool(
                fact_rows,
                local_fact_buckets,
                restamped,
            ),
            restamped,
        )
        _assert_local_fact_vintages(candidate_facts, restamped, rosters)
        try:
            registry = compile_ledger_target_references(
                candidate_facts,
                [restamped],
                country="uk",
            )
        except ValueError as error:
            unsupported.append(
                {
                    "name": reference.name,
                    "period": target_period,
                    "reason": str(error),
                }
            )
        else:
            compiled.extend(registry.specs)
    return UKLedgerTargetCompilation(
        TargetRegistry(compiled, country="uk"),
        tuple(unsupported),
    )


def _assert_local_fact_vintages(
    facts: tuple[Mapping[str, Any], ...],
    reference: LedgerTargetReference,
    rosters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Refuse a matched fact whose boundary vintage is not the declared one.

    The crosswalk declares the boundary vintage per level (per code prefix at
    local-authority level, where the nations publish on different frames).
    Before this check the declaration was interpolated into error text only;
    now a fact on the wrong boundary set fails the compile, by name, instead
    of binding a value across vintages (PR #795 review, closing note).
    """

    selector = reference.ledger_selector
    level = str(selector.get("geography_level") or "")
    expected = rosters.get(level, {}).get("expected_vintage", "")
    if not expected:
        # A level the crosswalk declares no vintage for cannot be proven onto
        # any boundary frame (re-review finding 3: the silent escapes are the
        # cases the gate most needs to catch).
        raise ValueError(
            f"UK local target reference {reference.name!r} is at level "
            f"{level!r}, which declares no expected boundary vintage in the "
            "crosswalk."
        )
    for fact in facts:
        geography = fact.get("geography")
        if not isinstance(geography, Mapping):
            continue
        vintage = str(geography.get("vintage") or "")
        code = str(geography.get("id") or "")
        if isinstance(expected, Mapping):
            wanted = expected.get(code[:1])
            if wanted is None:
                raise ValueError(
                    f"UK local target reference {reference.name!r} matched a "
                    f"fact at {code!r}, whose prefix has no declared boundary "
                    f"vintage in the crosswalk for level {level!r}."
                )
        else:
            wanted = expected
        accepted = (
            {str(wanted)} if isinstance(wanted, str) else {str(v) for v in wanted}
        )
        if not vintage:
            raise ValueError(
                f"UK local target reference {reference.name!r} matched a fact "
                f"at {code!r} that declares no boundary vintage; the gate "
                "exists to prove the frame, and an unstamped fact is the case "
                "it most needs to catch."
            )
        if vintage not in accepted:
            raise ValueError(
                f"UK local target reference {reference.name!r} matched a fact "
                f"at {code!r} with boundary vintage {vintage!r}; the crosswalk "
                f"accepts {sorted(accepted)} for level {level!r}."
            )


def _candidate_facts_for_reference(
    facts: tuple[Mapping[str, Any], ...],
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, Any], ...]:
    if not reference.ledger_selector:
        return facts
    return tuple(
        fact
        for fact in facts
        if _fact_matches_selector(fact, reference.ledger_selector)
    )


def _local_fact_buckets(
    facts: tuple[Mapping[str, Any], ...],
) -> dict[tuple[str, str], tuple[Mapping[str, Any], ...]]:
    materialized: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for fact in facts:
        key = _local_fact_geography_key(fact)
        if key is not None:
            materialized.setdefault(key, []).append(fact)
    return {key: tuple(rows) for key, rows in materialized.items()}


def _local_candidate_fact_pool(
    facts: tuple[Mapping[str, Any], ...],
    buckets: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]],
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, Any], ...]:
    selector = reference.ledger_selector
    geography_level = selector.get("geography_level")
    geography_id = selector.get("geography_id")
    if not isinstance(geography_level, str) or not isinstance(geography_id, str):
        return facts
    return buckets.get((geography_level, geography_id), ())


def _local_fact_geography_key(fact: Mapping[str, Any]) -> tuple[str, str] | None:
    geography = fact.get("geography")
    if not isinstance(geography, Mapping):
        return None
    level = geography.get("level")
    geography_id = geography.get("id")
    if not isinstance(level, str) or not isinstance(geography_id, str):
        return None
    if not level or not geography_id:
        return None
    return (level, geography_id)


def _local_crosswalk_rosters(
    crosswalk: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    levels = crosswalk.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("UK local area crosswalk must expose levels.")
    rosters: dict[str, dict[str, Any]] = {}
    for level, payload in levels.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"UK local area crosswalk level {level!r} is invalid.")
        area_ids = payload.get("area_ids")
        if not isinstance(area_ids, list) or not area_ids:
            raise ValueError(
                f"UK local area crosswalk level {level!r} must expose area_ids."
            )
        rosters[str(level)] = {
            "area_ids": frozenset(str(area_id) for area_id in area_ids),
            "expected_vintage": payload.get("expected_vintage", ""),
        }
    return rosters


def _assert_household_type_bindings_declared(
    contract: Mapping[str, Mapping[str, Any]],
) -> None:
    """Every ``ons_household_type`` condition names a declared frame value.

    The ten ONS household-composition rows bind on the frs_relationships
    stage's household column (microcosm#791), whose domain is Chronicle's
    ``ons.household_type`` value ids. A condition value outside that domain
    would match no household and surface only as a zero-support refusal deep
    in the solve; refusing at compile time names the row instead. The ten
    rows must also cover the domain exactly once, or the partition the
    census bridge reconciles against is no longer a partition.
    """

    from microcosm.build.uk_runtime.frs_relationships import (
        CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS,
    )

    domain = set(CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS)
    covered: dict[str, str] = {}
    for target_id, target in contract.items():
        binding = target.get("bindings", {}).get("policyengine", {})
        for field in ("filters", "household_conditions"):
            for predicate in binding.get(field, ()):
                if predicate.get("variable") != "ons_household_type":
                    continue
                value = predicate.get("value")
                if predicate.get("operator", "==") != "==" or value not in domain:
                    raise ValueError(
                        f"UK target {target_id!r} conditions on ons_household_type "
                        f"with {predicate!r}; the declared values are "
                        f"{sorted(domain)} under '=='."
                    )
                if target.get("family") == "ons_household_composition":
                    if value in covered:
                        raise ValueError(
                            f"UK targets {covered[value]!r} and {target_id!r} both "
                            f"bind ons_household_type == {value!r}."
                        )
                    covered[value] = target_id
    missing = domain - set(covered)
    if covered and missing:
        raise ValueError(
            "UK ons_household_composition rows leave declared household-type "
            f"value(s) unbound: {sorted(missing)}."
        )


def _assert_local_reference_in_crosswalk(
    reference: LedgerTargetReference,
    rosters: Mapping[str, Mapping[str, Any]],
) -> None:
    selector = reference.ledger_selector
    geography_level = str(selector.get("geography_level") or "")
    geography_id = str(selector.get("geography_id") or "")
    if not geography_level or not geography_id:
        raise ValueError(
            f"UK local target reference {reference.name!r} must pin "
            "geography_level and geography_id."
        )
    roster = rosters.get(geography_level)
    if roster is None:
        raise ValueError(
            f"UK local target reference {reference.name!r} uses unknown "
            f"geography level {geography_level!r}."
        )
    if geography_id not in roster["area_ids"]:
        raise ValueError(
            f"UK local target reference {reference.name!r} uses geography id "
            f"{geography_id!r} outside the {geography_level!r} roster for "
            f"expected vintage {roster['expected_vintage']!r}."
        )


def materialize_uk_ledger_targets(
    adapter: Any,
    registry: TargetRegistry,
    *,
    period: int | str,
    band_edge_registry: TargetRegistry | None = None,
) -> TargetMaterializationResult:
    """Materialize compiled UK Ledger target bindings on an adapter."""

    contract = _uk_contract_targets()
    return materialize_target_bindings(
        adapter,
        registry,
        contract,
        period=period,
        providers={
            "parameter_gated_threshold": _uk_parameter_gated_threshold,
            "baseline_flag_crosstab": _uk_baseline_flag_crosstab,
            "input_substitution_counterfactual": _uk_input_substitution,
        },
        band_edge_registry=band_edge_registry,
    )


#: Published-fact reductions rewritten to the internal reduction that carries
#: the same meaning on our frame. Facts keep the semantics of the source that
#: published them; translating those onto the model's own concepts is our job,
#: and a fact we cannot phrase internally gets translated and recorded, never
#: dropped.
#:
#: ``any_child_under`` (DWP Stat-Xplore, Scottish UC households with a child
#: under 1) names a dependent-child concept the model does not carry. There is
#: no ``is_child`` column because the model has no need of one: dependency is
#: derived from age where it is wanted. So "any child under N" is exactly "any
#: person aged under N" here, and the condition already supplies the age bound.
#: The rewrite is declared rather than aliased at the call site so it stays
#: greppable, testable, and visible in review.
UK_TRANSLATED_HOUSEHOLD_REDUCTIONS: Mapping[str, str] = {
    "any_child_under": "any",
}


class UKFrameTargetAdapter:
    """Frame-backed target materialization adapter for UK calibration stages."""

    def __init__(self, frame: Frame):
        self.frame = frame
        self.tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        self.link_tables = {name: frame.link(name).copy() for name in frame.links}

    def has_column(self, entity: str, variable: str) -> bool:
        return variable in self.tables[entity]

    def column(self, entity: str, variable: str) -> np.ndarray:
        table = self.tables[entity]
        count_variables = {
            "person_count",
            "household_count",
            "benunit_count",
            f"{entity}_count",
        }
        if variable in count_variables:
            return np.ones(len(table), dtype=float)
        if variable not in table:
            raise KeyError(f"{entity}.{variable}")
        return np.asarray(table[variable])

    def set_column(self, entity: str, variable: str, values: object) -> None:
        self.tables[entity][variable] = np.asarray(values)

    def parameter(self, parameter: str, period: int | str) -> float:
        if parameter == "gov.hmrc.cgt.annual_exempt_amount":
            return uk_cgt_annual_exempt_amount(period)
        raise KeyError(parameter)

    def counterfactual_delta(
        self,
        binding: Mapping[str, Any],
        period: int | str,
    ) -> np.ndarray:
        del period
        entity = str(binding.get("from_entity") or "person")
        metric_name = str(binding.get("metric_name") or "")
        if metric_name and metric_name in self.tables[entity]:
            return np.asarray(self.tables[entity][metric_name], dtype=float)
        raise ValueError(
            f"frame does not carry precomputed counterfactual delta {metric_name!r}"
        )

    def _household_ids_for(self, entity: str) -> Any:
        """The household each row of ``entity`` belongs to, row-aligned."""

        source = self.tables[entity]
        if entity == "household":
            return source["household_id"]
        if entity == "person":
            # People sit directly in a household; only group entities need the
            # membership lookup below, whose column would be the nonexistent
            # "person_person_id" here.
            return source["person_household_id"]
        people = self.tables["person"]
        entity_membership = f"person_{entity}_id"
        if entity_membership not in people:
            raise KeyError(entity_membership)
        group_to_household = (
            people[[entity_membership, "person_household_id"]]
            .drop_duplicates()
            .set_index(entity_membership)["person_household_id"]
        )
        if group_to_household.index.has_duplicates:
            raise ValueError(f"UK {entity} groups span multiple households.")
        return source[f"{entity}_id"].map(group_to_household)

    def entity_reduction(self, reduction: Mapping[str, Any]) -> np.ndarray:
        """Reduce a member-level variable to a household-grain value.

        The numeric sibling of :meth:`household_condition`: where that answers
        "does this household satisfy the predicate", this answers "what does
        this household's members sum to". A count target declared over a
        boolean member variable — the number of children, not whether the
        household has any — is only honest through this path; collapsing the
        boolean with ``any`` would publish an indicator against a count.
        """

        entity = str(reduction.get("entity") or "person")
        source = self.tables[entity]
        variable = str(reduction["variable"])
        if variable not in source:
            raise KeyError(f"{entity}.{variable}")
        household_ids = self._household_ids_for(entity)
        reduce = str(reduction.get("reduce") or "sum")
        values = source[variable]
        if reduce == "sum":
            aggregate = values.astype(float).groupby(household_ids).sum()
        elif reduce == "count":
            aggregate = values.groupby(household_ids).count()
        else:
            raise ValueError(f"Unsupported UK entity reduction {reduce!r}.")
        households = self.tables["household"]
        return np.asarray(
            households["household_id"].map(aggregate).fillna(0.0),
            dtype=float,
        )

    def household_condition(self, condition: Mapping[str, Any]) -> np.ndarray:
        """Evaluate a household predicate, optionally project it to its members.

        A benefit-unit target can share its dwelling's geography while keeping
        claimant/family filters at benefit-unit grain. ``map_to`` is explicit;
        legacy household conditions keep their household-aligned result.
        """
        entity = str(condition.get("entity") or "household")
        source = self.tables[entity]
        household_ids = self._household_ids_for(entity)

        published = str(condition["reduce"])
        reduce = UK_TRANSLATED_HOUSEHOLD_REDUCTIONS.get(published, published)
        variable = str(condition["variable"])
        if reduce == "any":
            matched = _compare_series(source[variable], condition)
            aggregate = matched.groupby(household_ids).any().astype(float)
            expected = {"operator": "==", "value": True}
        elif reduce == "sum":
            aggregate = source[variable].groupby(household_ids).sum()
            expected = condition
        elif reduce == "count":
            aggregate = source[variable].groupby(household_ids).count()
            expected = condition
        else:
            raise ValueError(
                f"Unsupported UK household reduction {published!r}."
                if published == reduce
                else f"Unsupported UK household reduction {published!r} "
                f"(translated to {reduce!r})."
            )

        households = self.tables["household"]
        ids = households["household_id"]
        matched = _compare_series(ids.map(aggregate).fillna(0.0), expected)
        map_to = str(condition.get("map_to") or "household")
        if map_to != "household":
            if map_to not in {"person", "benunit"}:
                raise ValueError(
                    f"Unsupported UK household condition map_to {map_to!r}."
                )
            matched = self._household_ids_for(map_to).map(
                pd.Series(matched.to_numpy(), index=ids)
            )
            if matched.isna().any():
                raise ValueError(
                    "UK household condition has unmatched household links."
                )
        return np.asarray(matched, dtype=bool)

    def to_frame(self) -> Frame:
        tables = {**self.tables, **self.link_tables}
        weights = {
            entity: self.frame.weights_for(entity)
            for entity in self.frame.weighted_entities
        }
        return Frame(
            tables,
            self.frame.schema,
            weights,
            self.frame.strata,
            mass_log=self.frame.mass_log,
            metadata=self.frame.metadata,
        )


@lru_cache(maxsize=2)
def _uk_contract_targets(
    *,
    national_only: bool = True,
) -> dict[str, Mapping[str, Any]]:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_POPULATION_TARGETS_RESOURCE)
        .read_text()
    )
    contract = json.loads(payload)
    _warn_on_undeclared_geography(contract["targets"])
    return {
        target["target_id"]: target
        for target in contract["targets"]
        if not national_only
        or set(target.get("geography_levels") or ())
        <= UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS
    }


def _spec_geography(spec: TargetSpec) -> tuple[str, str]:
    """Resolve one compiled target's local or Ledger geography spelling."""

    metadata = spec.metadata

    def spelling(prefix: str, label: str) -> tuple[str, str] | None:
        level_key = f"{prefix}geography_level"
        id_key = f"{prefix}geography_id"
        if level_key not in metadata and id_key not in metadata:
            return None
        level = str(metadata.get(level_key) or "").strip()
        geography_id = str(metadata.get(id_key) or "").strip()
        if not level or not geography_id:
            raise ValueError(
                f"UK target {spec.name!r} has blank {label} geography "
                f"(level={level!r}, id={geography_id!r})."
            )
        return level, geography_id

    local = spelling("", "local")
    ledger = spelling("ledger_", "ledger")
    if local is not None and ledger is not None and local != ledger:
        raise ValueError(
            f"UK target {spec.name!r} geography spellings disagree: "
            f"local={local!r}, ledger={ledger!r}."
        )
    resolved = local or ledger
    if resolved is None:
        raise ValueError(
            f"UK target {spec.name!r} names no geography under either the "
            "local or Ledger metadata spelling."
        )

    level, geography_id = resolved
    if local is None or level in UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS:
        contract_target_id = str(
            metadata.get("contract_target_id", spec.name.split("@", 1)[0])
        )
        contract = _uk_contract_targets(national_only=False).get(contract_target_id)
        if contract is None:
            raise ValueError(
                f"UK target {spec.name!r} references unknown contract target "
                f"{contract_target_id!r}."
            )
        declared_levels = tuple(
            str(value).strip()
            for value in contract.get("geography_levels") or ()
            if str(value).strip()
        )
        if level not in declared_levels:
            raise ValueError(
                f"UK target {spec.name!r} resolved national geography level "
                f"{level!r}, which disagrees with contract target "
                f"{contract_target_id!r} levels {list(declared_levels)!r}."
            )
    return level, geography_id


def apply_uk_cross_grain_reconciliation(
    local_frame: pd.DataFrame,
    bound_higher_targets: Iterable[str],
    *,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the standing UK rule to a bound mixed-grain target surface.

    Increment #762 may extend the grains in the UK rowwise solve only through
    this front door, so detection, reconciliation, and the manifest receipt
    cannot be bypassed.
    """

    licences = (
        _uk_licensed_empty_legs_from_membership(
            load_uk_local_target_reference_membership()
        )
        if licensed_empty_legs is None
        else licensed_empty_legs
    )
    return apply_cross_grain_reconciliation(
        local_frame,
        bound_higher_targets,
        _uk_contract_targets(national_only=False),
        UK_CROSS_GRAIN_RULE,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        licensed_empty_legs=licences,
    )


#: The Ledger concept the A15 Chronicle census-household uprating stands on:
#: ONS "Families and households in the UK", Table 5 all-households row, UK,
#: calendar year.
UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT = "ons.households_total"
UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY = "K02000001"


def uk_ledger_households_total(
    facts: Iterable[Mapping[str, Any]],
    *,
    period: int | str,
) -> dict[str, Any]:
    """Select the published UK household total for ``period`` from the Ledger.

    Exactly one fact must carry the ``ons.households_total`` concept at the
    UK country geography for the calendar year ``period`` with no
    dimensions; zero or several fail closed by name, so an artifact that
    lacks the vintage cannot silently bind the census-vintage household rows.
    """

    target_period = int(period)
    matches: list[Mapping[str, Any]] = []
    for fact in facts:
        alignment = fact.get("concept_alignment")
        if not isinstance(alignment, Mapping):
            continue
        if alignment.get("canonical_concept") != UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT:
            continue
        geography = fact.get("geography")
        if not isinstance(geography, Mapping) or geography.get("id") != (
            UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY
        ):
            continue
        fact_period = fact.get("period")
        if not isinstance(fact_period, Mapping):
            continue
        if fact_period.get("type") != "calendar_year":
            continue
        try:
            if int(fact_period.get("value")) != target_period:
                continue
        except (TypeError, ValueError):
            continue
        if fact.get("dimensions"):
            continue
        matches.append(fact)
    if len(matches) != 1:
        raise ValueError(
            f"UK census household uprating needs exactly one Ledger fact for "
            f"{UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT!r} at "
            f"{UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY} for calendar year "
            f"{target_period}; found {len(matches)}."
        )
    fact = matches[0]
    value = float(fact.get("value"))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"UK census household uprating reference must be a positive finite "
            f"count, got {fact.get('value')!r}."
        )
    lineage = fact.get("lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}
    return {
        "concept": UK_LEDGER_HOUSEHOLDS_TOTAL_CONCEPT,
        "geography_id": UK_LEDGER_HOUSEHOLDS_TOTAL_GEOGRAPHY,
        "period": target_period,
        "value": value,
        "semantic_fact_key": str(fact.get("semantic_fact_key", "")),
        "aggregate_fact_key": str(fact.get("aggregate_fact_key", "")),
        "source_record_id": str(lineage.get("source_record_id", "")),
    }


def uk_census_household_uprating(
    local_registry: TargetRegistry,
    households_reference: Mapping[str, Any],
    *,
    period: int | str,
) -> dict[str, Any]:
    """Derive per-grain factors from compiled Chronicle census households."""

    reference_value = float(households_reference["value"])
    if int(households_reference.get("period", period)) != int(period):
        raise ValueError(
            "UK census household uprating reference period "
            f"{households_reference.get('period')!r} is not the calibration "
            f"period {period!r}."
        )
    grouped: dict[str, list[TargetSpec]] = {}
    for spec in local_registry.specs:
        if not spec.name.startswith(f"{UK_CENSUS_HOUSEHOLDS_TARGET_ID}@"):
            continue
        level, _ = _spec_geography(spec)
        if level not in {"constituency", "local_authority"}:
            raise ValueError(
                f"UK census household target {spec.name!r} has unsupported "
                f"geography level {level!r}."
            )
        from_period = spec.metadata.get("uprating_from_period")
        to_period = spec.metadata.get("uprating_to_period")
        try:
            held_to_period = int(to_period) == int(period)
            int(from_period)
        except (TypeError, ValueError):
            held_to_period = False
        if not held_to_period:
            raise ValueError(
                f"UK census household target {spec.name!r} must carry an "
                f"identity hold to period {period!r}."
            )
        grouped.setdefault(level, []).append(spec)
    if set(grouped) != {"constituency", "local_authority"}:
        raise ValueError(
            "UK census household uprating requires compiled constituency and "
            "local_authority cells."
        )
    grains: dict[str, dict[str, Any]] = {}
    for level, specs in sorted(grouped.items()):
        total = math.fsum(float(spec.value) for spec in specs)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"UK census household {level} total must be positive and finite."
            )
        factor = reference_value / total
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"UK census household {level} uprating factor is invalid: {factor!r}."
            )
        grains[level] = {
            "cells": len(specs),
            "census_households_total": total,
            "census_years": sorted(
                {int(spec.metadata["uprating_from_period"]) for spec in specs}
            ),
            "factor": factor,
        }
    return {
        "applied": True,
        "period": int(period),
        "reference": dict(households_reference),
        "grains": grains,
        "adjudication": (
            "microcosm#887 (per-grain Chronicle denominator supersedes #762 "
            "A15; A17 rule unchanged, factor moves from 1.0335759 to the "
            "LA-grain 1.0335595)"
        ),
    }


def uk_private_rent_mean_to_total(
    target_frame: pd.DataFrame,
    *,
    months: int | float = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compose PIPR monthly price levels into annual private-rent totals.

    The rowwise calibration surface is linear in household weights, so each
    ``rent/private_rent`` price level is multiplied by the matching authority's
    bound ``tenure/private_rent`` household count and the months in a year.
    """

    if target_frame.empty or "metric" not in target_frame.columns:
        return target_frame.copy(deep=True), {
            "applied": False,
            "reason": "no private_rent rows on the surface",
        }
    rent_positions = np.flatnonzero(
        target_frame["metric"].astype(str).to_numpy() == "rent/private_rent"
    )
    if not len(rent_positions):
        return target_frame.copy(deep=True), {
            "applied": False,
            "reason": "no private_rent rows on the surface",
        }

    tenure_positions_by_area = {
        str(target_frame.iloc[position]["area_code"]): position
        for position in np.flatnonzero(
            target_frame["metric"].astype(str).to_numpy() == "tenure/private_rent"
        )
    }
    rent_areas = [
        str(target_frame.iloc[position]["area_code"]) for position in rent_positions
    ]
    missing_areas = sorted(set(rent_areas) - set(tenure_positions_by_area))
    if missing_areas:
        raise ValueError(
            "private-rent price levels have no tenure/private_rent row for "
            f"area(s) {missing_areas}."
        )

    def _positive_finite(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) and numeric > 0 else None

    invalid_mean_areas = sorted(
        {
            str(target_frame.iloc[position]["area_code"])
            for position in rent_positions
            if _positive_finite(target_frame.iloc[position]["value"]) is None
        }
    )
    if invalid_mean_areas:
        raise ValueError(
            "private-rent monthly mean must be positive and finite for area(s) "
            f"{invalid_mean_areas}."
        )
    invalid_tenure_areas = sorted(
        {
            area_code
            for area_code in rent_areas
            if _positive_finite(
                target_frame.iloc[tenure_positions_by_area[area_code]]["value"]
            )
            is None
        }
    )
    if invalid_tenure_areas:
        raise ValueError(
            "tenure/private_rent value must be positive and finite for area(s) "
            f"{invalid_tenure_areas}."
        )

    composed = target_frame.copy(deep=True)
    if "metadata" not in composed.columns:
        composed["metadata"] = None
    cells_detail: list[dict[str, Any]] = []
    for position, area_code in zip(rent_positions, rent_areas, strict=True):
        tenure_position = tenure_positions_by_area[area_code]
        mean = float(target_frame.iloc[position]["value"])
        renter_households = float(target_frame.iloc[tenure_position]["value"])
        total = float(months) * mean * renter_households
        existing_metadata = target_frame.iloc[position].get("metadata")
        metadata = (
            dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}
        )
        metadata.update(
            {
                "price_level_mean_monthly": mean,
                "renter_households": renter_households,
                "renter_households_target_name": str(
                    target_frame.iloc[tenure_position]["target_name"]
                ),
            }
        )
        composed.iat[int(position), composed.columns.get_loc("value")] = total
        composed.iat[int(position), composed.columns.get_loc("metadata")] = metadata
        cells_detail.append(
            {
                "area_code": area_code,
                "mean_monthly_rent": mean,
                "renter_households": renter_households,
                "total": total,
            }
        )
    return composed, {
        "applied": True,
        "months": months,
        "cells": len(cells_detail),
        "adjudication": "microcosm#355 (ruling 2026-09-08)",
        "reason": (
            "PIPR supplies monthly private-rent price levels while the bound "
            "metric is an annual weighted total; compose each mean with the "
            "same authority's A17-uprated private-renter household count."
        ),
        "price_level_source": ("ons_pipr_private_rents calendar_year_average 2025"),
        "renter_count_source": "ons.tenure.private_rent (A17-uprated)",
        "cells_detail": cells_detail,
    }


def _is_census_vintage_hold(
    metadata: Mapping[str, Any],
    period: int | str,
    *,
    census_years: frozenset[int] | None = None,
) -> bool:
    """True for a compiled reference held from its grain's census vintage."""

    years = frozenset({2021, 2022}) if census_years is None else census_years
    from_period = metadata.get("uprating_from_period")
    to_period = metadata.get("uprating_to_period")
    if from_period is None or to_period is None:
        return False
    try:
        return int(from_period) in years and int(to_period) == int(period)
    except (TypeError, ValueError):
        return False


def uk_local_target_surface(
    local_registry: TargetRegistry,
    *,
    bound_national_target_ids: Iterable[str],
    period: int | str,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
    census_household_uprating: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble and reconcile the present-cell UK local target surface.

    ``census_household_uprating`` is the #887 per-grain receipt. Eligible
    census-household and tenure holds take their grain's factor.
    """

    if census_household_uprating is None:
        uprating_receipt: dict[str, Any] = {
            "applied": False,
            "grains": {},
            "reason": "no census household uprating receipt supplied.",
        }
    elif not census_household_uprating.get("applied"):
        uprating_receipt = dict(census_household_uprating)
        uprating_receipt["applied"] = False
        uprating_receipt.setdefault("grains", {})
    else:
        uprating_receipt = dict(census_household_uprating)
        grains = uprating_receipt.get("grains")
        if not isinstance(grains, Mapping):
            raise ValueError("census household uprating grains must be a mapping.")
        for level, grain in grains.items():
            factor = float(grain["factor"])
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(
                    f"census household {level} uprating factor is invalid: {factor!r}."
                )
    # microcosm#762 A17 (ruling 2026-09-03): census tenure cells share the
    # Chronicle census-household universe at the same grain. Identity-held
    # household and tenure cells take that grain's factor, preserving the
    # published tenure shares. A tenure cell compiled directly at the
    # calibration period carries no hold and is never touched.
    tenure_uprated: dict[str, int] = {}
    tenure_holds: list[dict[str, Any]] = []
    household_uprated: dict[str, int] = {}
    household_holds: list[dict[str, Any]] = []

    level_to_area_type = {
        level: area_type
        for area_type, level in AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL.items()
    }
    target_id_to_metric = {
        target_id: metric for metric, target_id in _uk_local_metric_target_ids().items()
    }
    output_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    national_control_groups: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    for spec in local_registry.specs:
        geography_level, geography_id = _spec_geography(spec)
        contract_target_id = str(
            spec.metadata.get("contract_target_id", spec.name.split("@", 1)[0])
        )
        if geography_level in level_to_area_type:
            area_type = level_to_area_type[geography_level]
            metric = target_id_to_metric.get(contract_target_id)
            if metric is None:
                raise ValueError(
                    "UK local target surface cannot map contract target "
                    f"{contract_target_id!r} to a PolicyEngine metric."
                )
            allowed = set(metric_names(area_type))
            if metric not in allowed:
                raise ValueError(
                    f"UK local target surface metric {metric!r} is not declared "
                    f"for area_type {area_type!r}."
                )
            output_position = len(output_rows)
            value = float(spec.value)
            is_households = contract_target_id == UK_CENSUS_HOUSEHOLDS_TARGET_ID
            is_tenure = contract_target_id.startswith("ons.tenure.")
            if is_households or is_tenure:
                from_period = spec.metadata.get("uprating_from_period")
                to_period = spec.metadata.get("uprating_to_period")
                grain = uprating_receipt.get("grains", {}).get(geography_level)
                years = (
                    frozenset(grain.get("census_years", ())) if grain else frozenset()
                )
                attempted = from_period is not None or to_period is not None
                eligible = attempted and _is_census_vintage_hold(
                    spec.metadata, period, census_years=years
                )
                factor = float(grain["factor"]) if grain else 1.0
                applied = bool(
                    eligible and grain is not None and uprating_receipt.get("applied")
                )
                if is_households and uprating_receipt.get("applied") and not applied:
                    raise ValueError(
                        f"UK census household denominator {spec.name!r} is not "
                        f"eligible for its {geography_level} grain factor."
                    )
                if applied:
                    value *= factor
                    applied_by_vintage = (
                        household_uprated if is_households else tenure_uprated
                    )
                    applied_by_vintage[str(from_period)] = (
                        applied_by_vintage.get(str(from_period), 0) + 1
                    )
                    reason = "census_vintage_hold_uprated"
                elif not attempted:
                    reason = "no_identity_hold"
                elif grain is None:
                    reason = "missing_grain_uprating"
                elif not eligible:
                    reason = "hold_not_from_grain_census_vintage_or_wrong_period"
                else:
                    reason = "census_household_uprating_not_applied"
                hold_rows = household_holds if is_households else tenure_holds
                hold_rows.append(
                    {
                        "target_name": spec.name,
                        "geography_level": geography_level,
                        "from_period": from_period,
                        "to_period": to_period,
                        "attempted": attempted,
                        "eligible": eligible,
                        "applied": applied,
                        "skipped": not applied,
                        "reason": reason,
                    }
                )
            output_rows.append(
                {
                    "area_type": area_type,
                    "area_code": geography_id,
                    "metric": metric,
                    "value": value,
                    "target_name": spec.name,
                    "family": family_for_metric(metric),
                    "source_family": spec.family,
                    "source": spec.source,
                    "period": period,
                    "contract_target_id": contract_target_id,
                    "hierarchy": spec.hierarchy,
                }
            )
            reconciliation_rows.append(
                {
                    "grain": area_type,
                    "geography_id": geography_id,
                    "target_id": f"contract:{contract_target_id}",
                    "value": value,
                    "_output_position": output_position,
                }
            )
        elif geography_level in UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS:
            value = float(spec.value)
            if not np.isfinite(value):
                raise ValueError(
                    f"UK national target cell {spec.name!r} has non-finite "
                    f"value {value!r}."
                )
            national_control_groups.setdefault(
                (contract_target_id, geography_id), []
            ).append((spec.name, geography_level, value))
        else:
            raise ValueError(
                f"UK target {spec.name!r} names unsupported geography_level "
                f"{geography_level!r}."
            )

    bridge_control_ids = {
        target_id
        for bridge in UK_CROSS_GRAIN_BRIDGES
        for target_id in bridge.higher_target_ids
    }
    fanout_target_ids = {
        target_id
        for (target_id, _), cells in national_control_groups.items()
        if len(cells) > 1 and target_id not in bridge_control_ids
    }
    fanout_targets_not_controls: list[dict[str, Any]] = []
    for (target_id, geography_id), cells in national_control_groups.items():
        cell_names = sorted(name for name, _, _ in cells)
        geography_levels = sorted({level for _, level, _ in cells})
        if len(geography_levels) != 1:
            raise ValueError(
                f"UK national target {target_id!r} at {geography_id!r} has "
                f"fan-out cells {cell_names} with mixed geography levels "
                f"{geography_levels}."
            )
        activated_sum = math.fsum(value for _, _, value in cells)
        if not math.isfinite(activated_sum):
            raise ValueError(
                f"UK national target {target_id!r} at {geography_id!r} has "
                f"fan-out cells {cell_names} with a non-finite summed total."
            )
        if target_id in bridge_control_ids and len(cells) > 1:
            raise ValueError(
                f"UK national target {target_id!r} is a cross-grain bridge control "
                f"but fans out into {len(cells)} cells {cell_names} at "
                f"{geography_id!r}; a bridge control must be one cell per geography."
            )
        if target_id in fanout_target_ids:
            if len(cells) == 1:
                # The target-id rule drops the target as a control at every
                # geography once it fans out at any; say so where it is a
                # single cell rather than dropping it silently.
                fanout_targets_not_controls.append(
                    {
                        "target_id": target_id,
                        "geography_id": geography_id,
                        "cells": 1,
                        "cell_names": cell_names,
                        "activated_sum": activated_sum,
                        "reason": (
                            "Single cell at this geography, but the target fans "
                            "out at another geography, so the target-id rule "
                            "drops it as a control everywhere."
                        ),
                    }
                )
            if len(cells) > 1:
                fanout_targets_not_controls.append(
                    {
                        "target_id": target_id,
                        "geography_id": geography_id,
                        "cells": len(cells),
                        "cell_names": cell_names,
                        "activated_sum": activated_sum,
                        "reason": (
                            "The activated cells are a band subset, so this "
                            "distribution is not a cross-grain control."
                        ),
                    }
                )
            continue
        reconciliation_rows.extend(
            {
                "grain": geography_level,
                "geography_id": geography_id,
                "target_id": f"contract:{target_id}",
                "value": value,
                "_output_position": None,
            }
            for _, geography_level, value in cells
        )

    bound_control_ids = tuple(
        str(target_id)
        for target_id in bound_national_target_ids
        if str(target_id) not in fanout_target_ids
    )

    reconciliation = pd.DataFrame(
        reconciliation_rows,
        columns=["grain", "geography_id", "target_id", "value", "_output_position"],
    )
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        reconciliation[["grain", "geography_id", "target_id", "value"]],
        bound_control_ids,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        licensed_empty_legs=licensed_empty_legs,
    )
    receipt["fanout_targets_not_controls"] = fanout_targets_not_controls

    def cell_receipt(
        holds: list[dict[str, Any]],
        uprated: Mapping[str, int],
    ) -> dict[str, Any]:
        return {
            "applied": bool(uprated),
            "cells": int(sum(uprated.values())),
            "total_cells": len(holds),
            "attempted_cells": sum(row["attempted"] for row in holds),
            "eligible_cells": sum(row["eligible"] for row in holds),
            "skipped_cells": sum(row["skipped"] for row in holds),
            "holds": holds,
            "by_census_vintage": dict(sorted(uprated.items())),
        }

    uprating_receipt["household_cells"] = cell_receipt(
        household_holds, household_uprated
    )
    uprating_receipt["tenure_cells"] = {
        **cell_receipt(tenure_holds, tenure_uprated),
        "adjudication": "microcosm#762 (A17, ruling 2026-09-03)",
        "reason": (
            "Census tenure cells held to the calibration period take the "
            "corresponding Chronicle census-household grain factor so their "
            "published shares remain unchanged."
        ),
    }
    receipt["census_household_uprating"] = uprating_receipt
    for position, value in enumerate(reconciled["value"].to_numpy(dtype=np.float64)):
        output_position = reconciliation.iloc[position]["_output_position"]
        if pd.notna(output_position):
            output_rows[int(output_position)]["value"] = float(value)
    surface, private_rent_receipt = uk_private_rent_mean_to_total(
        pd.DataFrame(output_rows)
    )
    receipt["private_rent_mean_to_total"] = private_rent_receipt
    return surface, receipt


def _validate_uk_cross_grain_declarations() -> None:
    contract = _uk_contract_targets(national_only=False)
    matched_sides: dict[str, str] = {}
    unknown: list[str] = []
    for bridge in UK_CROSS_GRAIN_BRIDGES:
        for target_id in bridge.higher_target_ids:
            if target_id not in contract:
                unknown.append(target_id)
        if not bridge.lower_side.startswith("contract:"):
            raise ValueError(
                f"UK cross-grain bridge {bridge.bridge_id!r} lower side must "
                f"start with 'contract:', got {bridge.lower_side!r}."
            )
        lower_target_id = bridge.lower_side.removeprefix("contract:")
        if lower_target_id not in contract:
            unknown.append(lower_target_id)
        for side in (*bridge.higher_target_ids, bridge.lower_side):
            canonical = side.removeprefix("contract:")
            existing = matched_sides.get(canonical)
            if existing is not None:
                raise ValueError(
                    f"UK cross-grain target {canonical!r} is covered by both "
                    f"{existing!r} and {bridge.bridge_id!r}."
                )
            matched_sides[canonical] = bridge.bridge_id
    if unknown:
        raise ValueError(
            "UK cross-grain bridge target id(s) are absent from the committed "
            f"contract: {sorted(set(unknown))}."
        )


def _warn_on_undeclared_geography(targets) -> None:
    # Ruling on PR #795 review finding 3: an absent geography_levels reads as
    # national by doctrine (the empty set is a subset of the national levels),
    # and the contract test requires every committed target to declare the
    # field -- so this warning only ever fires on a hand-built contract, where
    # a silent default into the national surface is worth a loud note.
    undeclared = [
        str(target.get("target_id", "<unknown>"))
        for target in targets
        if not target.get("geography_levels")
    ]
    if undeclared:
        import warnings

        warnings.warn(
            "UK population contract target(s) declare no geography_levels and "
            f"default to the national surface: {undeclared[:5]}",
            stacklevel=3,
        )


def _uk_parameter_gated_threshold(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import parameter_gated_threshold

    return parameter_gated_threshold(adapter, binding, period)


def _uk_baseline_flag_crosstab(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import baseline_flag_crosstab

    return baseline_flag_crosstab(adapter, binding, period)


def _uk_input_substitution(
    adapter: Any,
    binding: Mapping[str, Any],
    period: int | str,
) -> np.ndarray:
    from microcosm.build.target_materialization import input_substitution_counterfactual

    return input_substitution_counterfactual(adapter, binding, period)


def _compare_series(
    values: pd.Series,
    condition: Mapping[str, Any],
) -> pd.Series:
    operator = condition.get("operator")
    if operator is None:
        operator, expected = "==", condition["equals"]
    else:
        expected = condition["value"]
    if operator == "in":
        return values.isin(expected)
    operations = {
        "==": values.eq,
        "!=": values.ne,
        ">": values.gt,
        ">=": values.ge,
        "<": values.lt,
        "<=": values.le,
    }
    if operator not in operations:
        raise ValueError(f"Unsupported UK target condition operator {operator!r}.")
    return operations[operator](expected)


_validate_uk_cross_grain_declarations()
