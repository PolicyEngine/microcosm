"""Opt-in period aging for US dollar-amount calibration targets.

Populace calibrates the US 2024 weights against SOI TY2022/TY2023 dollar
levels applied at the 2024 build period *un-aged*. Because the calibration
hits those source-year levels almost exactly, simulated current-year (2025+)
income aggregates run systematically ~6-10% under current-year projections
(PolicyEngine/populace#212, #116: AGI $16.0T vs CBO TY2025 ~$17T+; income tax
$2.15T vs FY2025 receipts ~$2.4T). The residual is a period-vintage artifact,
not a support or weighting error.

This module ages dollar-amount targets from their *source* period to the
*build* period at compile time, using growth ratios computed from CBO
revenue-projection facts already present in the Ledger consumer feed. It never
hardcodes a numeric factor: every factor is a ratio of two source-published
CBO ``projected_amount`` facts, and each aged (or deliberately un-aged) target
records the fact lineage it used.

The schema home for the *fact-vs-computed* boundary is
PolicyEngine/ledger#71 (facts-only store; PolicyEngine-computed aged levels
live in Populace as a named, versioned aging implementation that consumes
growth-factor facts from Ledger). This module is that Populace-side
implementation: :data:`AGING_MODEL_ID`/:data:`AGING_MODEL_VERSION` are the
alignment declaration recorded on every aged target. Aging the values stays
opt-in, but the period contract itself is enforced:
:func:`enforce_period_contract` makes silently consuming an observation
dollar level at a period other than its fact period a hard error — the
populace#212 failure mode — unless the build explicitly waives it.

Factor policy (priority order):

1. **Matching CBO series.** A target whose measured concept corresponds to a
   CBO income-by-source row (AGI, wages, net capital gain, qualified
   dividends, net business income) is aged by that series' own projection
   ratio ``projected_amount(series, build) / projected_amount(series,
   source)``.
2. **Chained observed growth (v1.1).** When the CBO series does not cover
   the source year (the published projection detail starts at TY2023), the
   observed span is bridged with national SOI actuals for the same series
   and only the unobserved span uses the projection:
   ``soi(series, pivot) / soi(series, source) × cbo(series, build) /
   cbo(series, pivot)`` where the pivot is the earliest CBO year. Observed
   growth for observed years, projected growth only where nothing is
   observed.
3. **CBO AGI default.** Any other dollar amount is aged by the CBO AGI
   ratio, direct or chained by the national SOI AGI series.
4. **Counts stay raw.** Return/claim counts (``measure_mode ==
   "indicator_sum"``) and population counts are never aged here; a growing
   nominal aggregate does not imply a growing return count.

A target is only aged when both the build-year and the source-year projection
facts of the chosen series are present in the feed. Otherwise it is left at
its raw value with ``aging_factor_source="unavailable"`` so the un-aged state
is explicit in diagnostics rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from populace.calibrate import TargetRegistry, TargetSpec

__all__ = [
    "AGING_MODEL_ID",
    "AGING_MODEL_VERSION",
    "PeriodContractError",
    "PeriodContractViolation",
    "age_us_dollar_targets",
    "enforce_period_contract",
    "find_period_contract_violations",
]

# The named, versioned Populace-side alignment model (PolicyEngine/ledger#71).
# Bump the version whenever the factor policy changes; no Ledger fact and no
# consumer-artifact hash changes when it does — only Populace build outputs.
AGING_MODEL_ID = "cbo_growth_factor_aging"
AGING_MODEL_VERSION = "1.2.0"


# CBO income-by-source ``groupby_value_id`` for the AGI projection series. This
# is the priority-(b) default growth index for dollar amounts with no
# source-aligned CBO series.
_CBO_AGI_INCOME_SOURCE = "adjusted_gross_income"

# CBO revenue-projection facts are identified structurally in the consumer
# feed by source name + measure id + income-source groupby dimension.
_CBO_SOURCE_NAME = "cbo"
_CBO_PROJECTION_MEASURE_ID = "projected_amount"
_CBO_INCOME_SOURCE_GROUPBY_DIMENSION = "cbo.income_source"

# Priority-(a) map: SOI/direct target ``source_measure_id`` -> CBO
# income-by-source ``groupby_value_id``. Only concepts with a genuine
# same-concept CBO projection row belong here; everything else falls back to
# the AGI series. Qualified dividends map to the CBO qualified-dividend series;
# ordinary/taxable interest and non-qualified dividends have no clean
# single-series CBO analogue, so they take the AGI default.
_SOI_MEASURE_TO_CBO_INCOME_SOURCE: dict[str, str] = {
    "adjusted_gross_income": "adjusted_gross_income",
    "wages_salaries_amount": "wages_and_salaries",
    "net_capital_gains_amount": "net_capital_gain",
    "qualified_dividends_amount": "qualified_dividend_income",
    "schedule_c_income_amount": "net_business_income",
    "partnership_scorp_income_amount": "net_business_income",
}

# Chain sources for factor policy (2): the national SOI actual series that
# bridges source years the CBO projection detail does not cover. Keyed by CBO
# income source; the value names the canonical all-return national table and
# measure whose year-over-year growth is the observed segment of a chained
# factor. Series without a clean national SOI analogue fall back to the AGI
# chain via the existing AGI default.
_CBO_INCOME_SOURCE_TO_SOI_CHAIN: dict[str, tuple[str, str]] = {
    "adjusted_gross_income": ("table_1_1", "adjusted_gross_income"),
    "wages_and_salaries": ("table_1_4", "wages_salaries_amount"),
    "net_capital_gain": ("table_1_4", "net_capital_gains_amount"),
}
_SOI_SOURCE_NAME = "irs_soi"

# Direct-target roles (CBO-sourced calibration targets) map to their own CBO
# income source so a build-year CBO target is aged by its own series when the
# source period differs from the build period.
_TARGET_ROLE_TO_CBO_INCOME_SOURCE: dict[str, str] = {
    "cbo_adjusted_gross_income": "adjusted_gross_income",
    "cbo_wages_and_salaries": "wages_and_salaries",
    "cbo_qualified_dividend_income": "qualified_dividend_income",
    "cbo_net_capital_gain": "net_capital_gain",
    "cbo_net_business_income": "net_business_income",
    # BEA NIPA all-population income targets (nonfiler-inclusive) age by their
    # own CBO income-by-source series so a cy2023 source level is projected to
    # the build year on the matching concept, not the AGI default. The cy2024
    # NIPA wage level needs no aging (source == build period).
    "nipa_wages_and_salaries": "wages_and_salaries",
    "bea_state_wages": "wages_and_salaries",
    "nipa_proprietors_income": "net_business_income",
    # W-2 Box 7 social security tips (populace#451 item 3): tips are a wage
    # component, and the fact's TY2020 vintage predates the CBO projection
    # span, so the wages series' SOI actuals provide the chained bridge.
    "w2_social_security_tips_total": "wages_and_salaries",
}


def age_us_dollar_targets(
    registry: TargetRegistry,
    facts: tuple[object, ...],
    *,
    target_period: int | str,
) -> TargetRegistry:
    """Age dollar-amount targets from source period to the build period.

    Args:
        registry: The compiled target registry (already period-aligned
            within-surface by the SOI/EITC uprating passes).
        facts: The materialized Ledger consumer facts. CBO revenue-projection
            facts in this feed supply every growth ratio.
        target_period: The build period the targets are compiled for.

    Returns:
        A new registry. Ageable dollar targets whose source period differs
        from the build period are scaled by a CBO projection ratio and carry
        ``basis``/``source_period``/``aged_to``/``aging_factor``/
        ``aging_factor_source`` diagnostics. Every other spec is returned
        unchanged except for the same diagnostics recording that it was not
        aged (counts, same-period dollars) or could not be (missing facts).
    """

    projections = _cbo_projection_series(facts)
    chain_series = _soi_national_chain_series(facts)
    build_period_key = _period_year(target_period)
    specs: list[TargetSpec] = []
    for spec in registry.specs:
        specs.append(
            _age_spec(
                spec,
                projections=projections,
                chain_series=chain_series,
                target_period=target_period,
                build_period_key=build_period_key,
            )
        )
    return TargetRegistry(specs, country=registry.country)


@dataclass(frozen=True)
class PeriodContractViolation:
    """One dollar target consumed at a period its fact does not cover."""

    target_name: str
    fact_period: str
    build_period: str
    reason: str


class PeriodContractError(ValueError):
    """Raised when observation dollar levels would calibrate at the wrong period.

    This is the populace#212 guard, mirroring the Ledger consumption contract
    (PolicyEngine/ledger#71): a fact's value refers to its own reference
    period, and consuming it at any other period must be an explicit,
    declared transformation — never a silent default. Fix a violation by
    enabling compile-time aging (the ``cbo_growth_factor_aging`` model),
    supplying the missing CBO projection facts, or passing
    ``allow_unaged_dollar_targets=True`` to record a deliberate waiver.
    """

    def __init__(self, violations: tuple[PeriodContractViolation, ...]) -> None:
        self.violations = violations
        preview = "; ".join(
            f"{violation.target_name} (fact {violation.fact_period} != build "
            f"{violation.build_period}: {violation.reason})"
            for violation in violations[:5]
        )
        suffix = "" if len(violations) <= 5 else f"; +{len(violations) - 5} more"
        super().__init__(
            f"Period contract violation on {len(violations)} dollar "
            "target(s): observation facts cannot calibrate at a period other "
            "than their reference period without a declared alignment "
            "(PolicyEngine/ledger#71; populace#212). Enable age_targets, "
            "supply the missing CBO projection facts, or pass "
            f"allow_unaged_dollar_targets=True to waive explicitly. {preview}"
            f"{suffix}"
        )


def find_period_contract_violations(
    registry: TargetRegistry,
    *,
    target_period: int | str,
) -> tuple[PeriodContractViolation, ...]:
    """Find ageable observation dollar targets consumed at the wrong period.

    A spec violates the contract when it is an ageable nominal dollar amount
    (USD ``sum``, not already period-aligned within-surface), its backing
    fact is an observation (publisher projections are already statements
    about other periods), its fact year differs from the build year, and no
    alignment transformed it (``basis`` is not ``projection``).
    """

    build_year = _period_year(target_period)
    if build_year is None:
        return ()
    violations: list[PeriodContractViolation] = []
    for spec in registry.specs:
        if _not_ageable_reason(spec) is not None:
            continue
        metadata = spec.metadata
        if metadata.get("ledger_assertion") == "source_projection":
            continue
        if metadata.get("basis") == "projection":
            continue
        if metadata.get("period_contract_waiver"):
            continue
        fact_period = (
            metadata.get("source_period")
            or metadata.get("ledger_fact_period")
            or str(spec.period)
        )
        fact_year = _period_year(fact_period)
        if fact_year is None or fact_year == build_year:
            continue
        reason = (
            "aging_factor_unavailable"
            if metadata.get("aging_factor_source") == "unavailable"
            else "un_aged_dollar_target"
        )
        violations.append(
            PeriodContractViolation(
                target_name=spec.name,
                fact_period=str(fact_period),
                build_period=str(target_period),
                reason=reason,
            )
        )
    return tuple(violations)


def enforce_period_contract(
    registry: TargetRegistry,
    *,
    target_period: int | str,
    allow_unaged_dollar_targets: bool = False,
) -> TargetRegistry:
    """Raise on silent cross-period dollar consumption, or record a waiver.

    With ``allow_unaged_dollar_targets`` the violating specs keep their raw
    values but carry ``period_contract_waiver`` metadata so the waived state
    is auditable in diagnostics instead of invisible.
    """

    violations = find_period_contract_violations(
        registry,
        target_period=target_period,
    )
    if not violations:
        return registry
    if not allow_unaged_dollar_targets:
        raise PeriodContractError(violations)
    violating_names = {violation.target_name for violation in violations}
    specs = tuple(
        replace(
            spec,
            metadata={
                **dict(spec.metadata),
                "period_contract_waiver": "allow_unaged_dollar_targets",
            },
        )
        if spec.name in violating_names
        else spec
        for spec in registry.specs
    )
    return TargetRegistry(specs, country=registry.country)


def _age_spec(
    spec: TargetSpec,
    *,
    projections: dict[str, dict[int, tuple[float, str]]],
    chain_series: dict[str, dict[int, tuple[float, str]]],
    target_period: int | str,
    build_period_key: int | None,
) -> TargetSpec:
    aged_to = str(target_period)
    not_ageable_reason = _not_ageable_reason(spec)
    if not_ageable_reason is not None:
        # Counts, non-USD rows, and rows already period-aligned upstream are
        # never aged here; record the decision so the surface is auditable,
        # but do not touch the value.
        return _with_aging_metadata(
            spec,
            basis="fact",
            source_period=spec.metadata.get("source_period", ""),
            aged_to=aged_to,
            aging_factor=1.0,
            aging_factor_source=not_ageable_reason,
        )

    # An uprated row's value sits wherever the uprating pass landed it
    # (``uprating_to_period`` — the rebase control's period, not the build
    # period); age the remaining links from there. A row uprated all the way
    # to the build period falls through to source_equals_build with factor
    # one, and a row stopped at an earlier control (the Build N
    # net-capital-gains cross-vintage defect) receives exactly the missing
    # links. Unrebased rows age from their source period as before.
    source_period = str(
        spec.metadata.get("uprating_to_period")
        or spec.metadata.get("source_period", "")
        if "uprating_factor" in spec.metadata
        else spec.metadata.get("source_period", "")
    )
    source_period_key = _period_year(source_period)
    if "uprating_factor" in spec.metadata and source_period_key is None:
        # The within-surface uprating stamps are populace-authored; a
        # malformed effective period is a code bug and must never fail open
        # into a silent factor-one pass (PR #488 review finding 1).
        raise ValueError(
            f"Uprated target {spec.name!r} carries an unparseable "
            f"uprating_to_period {source_period!r}."
        )
    if (
        source_period_key is None
        or build_period_key is None
        or source_period_key == build_period_key
    ):
        # Nothing to age: source already equals build (or periods are not
        # comparable). Leave raw; basis stays "fact".
        return _with_aging_metadata(
            spec,
            basis="fact",
            source_period=source_period,
            aged_to=aged_to,
            aging_factor=1.0,
            aging_factor_source="source_equals_build"
            if source_period_key == build_period_key
            else "unavailable",
        )

    income_source = _cbo_income_source_for_spec(spec)
    factor_result = _projection_factor(
        projections,
        chain_series,
        income_source=income_source,
        build_period_key=build_period_key,
        source_period_key=source_period_key,
    )
    if factor_result is None and income_source != _CBO_AGI_INCOME_SOURCE:
        # Fall back to the CBO AGI growth ratio (direct or chained) when the
        # source-aligned series is unavailable for one of the two periods.
        factor_result = _projection_factor(
            projections,
            chain_series,
            income_source=_CBO_AGI_INCOME_SOURCE,
            build_period_key=build_period_key,
            source_period_key=source_period_key,
        )
    if factor_result is None:
        # No usable CBO projection pair: keep the raw source-year level but
        # make the un-aged state explicit (the ledger#71 lesson).
        return _with_aging_metadata(
            spec,
            basis="fact",
            source_period=source_period,
            aged_to=aged_to,
            aging_factor=1.0,
            aging_factor_source="unavailable",
        )

    factor, factor_source = factor_result
    aged = _with_aging_metadata(
        replace(spec, value=spec.value * factor),
        basis="projection",
        source_period=source_period,
        aged_to=aged_to,
        aging_factor=factor,
        aging_factor_source=factor_source,
    )
    # The alignment declaration: which named, versioned model computed this
    # level (PolicyEngine/ledger#71 — the declaration Ledger records but
    # never executes).
    return replace(
        aged,
        metadata={
            **dict(aged.metadata),
            "alignment_model_id": AGING_MODEL_ID,
            "alignment_model_version": AGING_MODEL_VERSION,
        },
    )


def _not_ageable_reason(spec: TargetSpec) -> str | None:
    """Why a spec is not an ageable nominal dollar amount, or ``None``.

    Ageable rows are USD ``sum`` measures. Counts (``indicator_sum``) stay
    raw. Rows rebased within-surface by an uprating pass carry
    ``uprating_factor`` and are aged FROM their control period (recorded as
    ``uprating_index_source_period``), never from the original source period
    — the rebase covered source->control, aging covers control->build, so no
    link is double-counted and none is dropped (the Build N net-capital-gains
    cross-vintage defect: the states' chain stopped at the TY2023 control
    while the national row continued to TY2024). The returned string is the
    ``aging_factor_source`` diagnostic recorded for the skip.
    """

    metadata = spec.metadata
    if metadata.get("measure_mode") != "sum":
        return "not_dollar_amount"
    unit = metadata.get("ledger_measure_unit", "")
    if unit and unit != "usd":
        return "not_usd_unit"
    if metadata.get("ledger_assertion") == "source_projection":
        # A publisher projection is consumed at its published level whatever
        # period it describes; re-projecting it here would silently compound
        # models. Mirrors the period-contract exemption.
        return "source_projection_level"
    return None


def _cbo_income_source_for_spec(spec: TargetSpec) -> str:
    """The CBO income-by-source series used to age ``spec``.

    Priority (a): a source-aligned CBO series when the target's measured
    concept has one. Priority (b): the CBO AGI series otherwise.
    """

    target_role = spec.metadata.get("target_role", "")
    direct = _TARGET_ROLE_TO_CBO_INCOME_SOURCE.get(target_role)
    if direct is not None:
        return direct
    source_measure_id = spec.metadata.get("source_measure_id", "")
    return _SOI_MEASURE_TO_CBO_INCOME_SOURCE.get(
        source_measure_id, _CBO_AGI_INCOME_SOURCE
    )


def _projection_factor(
    projections: dict[str, dict[int, tuple[float, str]]],
    chain_series: dict[str, dict[int, tuple[float, str]]],
    *,
    income_source: str,
    build_period_key: int,
    source_period_key: int,
) -> tuple[float, str] | None:
    """The growth factor for one series, direct or chained, if usable.

    Direct: ``projected(build) / projected(source)`` when the CBO series
    covers both years. Chained (factor policy 2): when the CBO series does
    not reach back to the source year, bridge the observed span with the
    series' national SOI actuals through the earliest CBO year (the pivot):
    ``soi(pivot) / soi(source) × projected(build) / projected(pivot)``.

    Returns the ratio and the factor's lineage — the build-year projection
    fact's source_record_id, prefixed ``chained:<soi pivot record>+`` for
    chained factors — or ``None`` when no usable pair exists.
    """

    series = projections.get(income_source)
    if not series:
        return None
    build = series.get(build_period_key)
    if build is None:
        return None
    build_value, build_record_id = build
    if build_value <= 0:
        return None

    source = series.get(source_period_key)
    if source is not None:
        source_value, _ = source
        if source_value <= 0:
            return None
        return build_value / source_value, build_record_id

    soi_series = chain_series.get(income_source)
    if not soi_series:
        return None
    pivot_year = min(series)
    if not (source_period_key < pivot_year <= build_period_key):
        return None
    pivot_projection = series.get(pivot_year)
    soi_source = soi_series.get(source_period_key)
    soi_pivot = soi_series.get(pivot_year)
    if pivot_projection is None or soi_source is None or soi_pivot is None:
        return None
    pivot_value, _ = pivot_projection
    soi_source_value, _ = soi_source
    soi_pivot_value, soi_pivot_record_id = soi_pivot
    if min(pivot_value, soi_source_value, soi_pivot_value) <= 0:
        return None
    factor = (soi_pivot_value / soi_source_value) * (build_value / pivot_value)
    return factor, f"chained:{soi_pivot_record_id}+{build_record_id}"


def _soi_national_chain_series(
    facts: tuple[object, ...],
) -> dict[str, dict[int, tuple[float, str]]]:
    """Index national SOI actuals usable as chained-factor bridges.

    Result maps CBO ``income_source`` -> ``{tax_year -> (value,
    source_record_id)}`` drawn from the canonical all-return national tables
    declared in ``_CBO_INCOME_SOURCE_TO_SOI_CHAIN``.
    """

    wanted = {
        (table, measure): income_source
        for income_source, (table, measure) in (_CBO_INCOME_SOURCE_TO_SOI_CHAIN.items())
    }
    series: dict[str, dict[int, tuple[float, str]]] = {}
    for fact in facts:
        if _source_name(fact) != _SOI_SOURCE_NAME:
            continue
        if _str_at(fact, "geography", "level") != "country":
            continue
        if _str_at(fact, "layout", "groupby_value_id") not in ("all", "total"):
            continue
        record_set_id = _str_at(fact, "layout", "record_set_id")
        segments = set(record_set_id.split("."))
        measure = _measure_id(fact)
        income_source = next(
            (
                mapped
                for (table, chain_measure), mapped in wanted.items()
                if table in segments and chain_measure == measure
            ),
            None,
        )
        if income_source is None:
            continue
        year = _period_year(_at(fact, "period", "value"))
        if year is None:
            continue
        value = _numeric_value(fact)
        record_id = _str_at(fact, "lineage", "source_record_id")
        existing = series.setdefault(income_source, {}).get(year)
        if existing is not None and existing != (value, record_id):
            raise ValueError(
                f"Conflicting national SOI chain facts for {income_source!r} "
                f"year {year}: {existing[1]!r} vs {record_id!r}. The chain "
                "bridge must be unambiguous; fix the feed."
            )
        series[income_source][year] = (value, record_id)
    return series


def _cbo_projection_series(
    facts: tuple[object, ...],
) -> dict[str, dict[int, tuple[float, str]]]:
    """Index CBO income-by-source projection facts by series and tax year.

    Result maps ``income_source -> {tax_year -> (value, source_record_id)}``.
    The source_record_id is retained so aged targets can cite the exact CBO
    fact whose value formed the numerator of their aging factor.
    """

    series: dict[str, dict[int, tuple[float, str]]] = {}
    for fact in facts:
        if not _is_cbo_income_source_projection_fact(fact):
            continue
        income_source = _str_at(fact, "layout", "groupby_value_id")
        if not income_source:
            continue
        year = _period_year(_at(fact, "period", "value"))
        if year is None:
            continue
        value = _numeric_value(fact)
        record_id = _str_at(fact, "lineage", "source_record_id")
        by_year = series.setdefault(income_source, {})
        existing = by_year.get(year)
        if existing is not None and existing != (value, record_id):
            raise ValueError(
                f"Conflicting CBO projection facts for {income_source!r} "
                f"year {year}: {existing[1]!r} vs {record_id!r}. Growth "
                "factors must come from one projection release; fix the feed."
            )
        by_year[year] = (value, record_id)
    return series


def _is_cbo_income_source_projection_fact(fact: object) -> bool:
    if _source_name(fact) != _CBO_SOURCE_NAME:
        return False
    if _measure_id(fact) != _CBO_PROJECTION_MEASURE_ID:
        return False
    groupby_dimension = _str_at(fact, "layout", "groupby_dimension")
    if groupby_dimension != _CBO_INCOME_SOURCE_GROUPBY_DIMENSION:
        return False
    if "revenue_projection" not in _str_at(fact, "layout", "record_set_id"):
        return False
    # Post-ledger#73 feeds type publisher projections explicitly. A row that
    # structurally looks like a CBO projection but is typed as an observation
    # is inconsistent data; refuse to use it as a growth factor. Feeds
    # predating the assertion field omit the key and stay eligible.
    assertion = _str_at(fact, "assertion")
    return assertion in ("", "source_projection")


def _with_aging_metadata(
    spec: TargetSpec,
    *,
    basis: str,
    source_period: str,
    aged_to: str,
    aging_factor: float,
    aging_factor_source: str,
) -> TargetSpec:
    metadata = {
        **dict(spec.metadata),
        "basis": basis,
        "aged_to": aged_to,
        "aging_factor": _format_float(aging_factor),
        "aging_factor_source": aging_factor_source,
    }
    if source_period:
        metadata["source_period"] = source_period
    return replace(spec, metadata=metadata)


def _period_year(value: object) -> int | None:
    """Extract a four-digit year from a period value/label, or ``None``.

    Handles bare ints and ``ty``/``cy``/``fy`` and ``tax_year_``-style labels.
    Monthly labels (``YYYY-MM``) reduce to their year for series matching.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    label = str(value).strip().lower().replace("-", "_")
    for prefix in ("month", "tax_year_", "calendar_year_", "fiscal_year_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    if label[:2] in {"ty", "cy", "fy"}:
        label = label[2:]
    head = label.split("_", maxsplit=1)[0]
    if head.isdigit() and len(head) == 4:
        return int(head)
    return None


def _format_float(value: float) -> str:
    return f"{value:.15g}"


# --- duck-typed Ledger fact accessors (mirrors fiscal_targets.py) ---------


def _at(obj: object, *path: str) -> object:
    current: object = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    return "" if value is None else str(value)


def _numeric_value(fact: object) -> float:
    value = _at(fact, "value")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _source_name(fact: object) -> str:
    return _str_at(fact, "observed_measure", "source_name") or _str_at(
        fact, "source", "source_name"
    )


def _measure_id(fact: object) -> str:
    return _str_at(fact, "observed_measure", "source_measure_id") or _str_at(
        fact, "layout", "measure_id"
    )
