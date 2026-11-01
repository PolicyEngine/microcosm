"""Ledger fact catalog adapters for Microcosm target profiles.

Ledger owns source-backed facts. Microcosm owns the active subset and the
model-variable mapping needed to compile a fact into a calibration target.
This module is intentionally duck-typed against the Ledger aggregate fact
schema so Microcosm can consume exported JSONL catalogs without importing the
Ledger implementation package at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any

from microcosm.build.ledger_artifact import LexicalFloat
from microcosm.calibrate import (
    CalibrationHierarchy,
    CalibrationHierarchySeed,
    HierarchyCategory,
    HierarchyDimension,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)
from microcosm.calibrate.geography_constants import UK_GEOGRAPHY_ID_TO_LABEL

SUPPORTED_LEDGER_AGGREGATIONS = frozenset(("sum",))
ALLOWED_ASSERTION_POLICIES = frozenset(("observed_only", "allow_source_projection"))
ALLOWED_PERIOD_MATCH_POLICIES = frozenset(
    ("latest_not_after", "exact", "source_window")
)
MONTHLY_WINDOW_OPERATIONS = frozenset(
    ("monthly_window_average", "monthly_window_sum_average")
)
#: A monthly window whose every point is a published count times a published
#: mean (a caseload and its mean weekly amount), scaled by the mean operand's
#: ``period_factor`` (52 turns a weekly mean into an annual amount) and
#: averaged over the declared months. The two operands name the count and mean
#: concepts, because publishers name the measures after the benefit
#: (``recipients``, ``mean_weekly_amount``) rather than ``*_count``/``*_mean``.
#: It is not one of ``MONTHLY_WINDOW_OPERATIONS``: those average a single
#: series, and this one pairs two.
MONTHLY_WINDOW_COUNT_X_MEAN = "monthly_window_count_x_mean"
#: ``rolled_forward_by_ratio``: an official estimate for an earlier period
#: (the base: one cell, or the sum of the cells the selector resolves at its
#: latest period) carried to the target period by the growth of a newer
#: official series: ``base * numerator / denominator``, where the numerator
#: is pinned to the target period and the denominator to the base's own
#: period (Wales's mid-2024 household estimate by its mid-2025/mid-2024
#: population, microcosm#1123). Unlike ``scaled_by_ratio`` the ratio may
#: exceed one; it is bounded to ``ROLLED_FORWARD_RATIO_BOUNDS`` so a
#: mis-pinned operand cannot pass for growth.
ROLLED_FORWARD_BY_RATIO = "rolled_forward_by_ratio"
ALLOCATED_RESIDUAL_BY_RATIO = "allocated_residual_by_ratio"
#: ``calendar_year_window_scaled_by_ratio``: the ``calendar_year_window`` of
#: the reference's own series, translated by the quotient of two facts that
#: each pin their own period and may come from another publisher: a subset's
#: share of the published universe, read where it is published (the
#: individuals' share of HMRC's SDLT receipts by property type restating the
#: OBR's SDLT lines for households, microcosm#1095). The quotient lies in
#: ``(0, 1]`` as for ``scaled_by_ratio``; the operands translate the window,
#: they do not join it.
CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO = "calendar_year_window_scaled_by_ratio"
ROLLED_FORWARD_OPERAND_ROLES = ("base", "numerator", "denominator")
ROLLED_FORWARD_RATIO_BOUNDS = (0.5, 2.0)
#: Ordered operand roles of ``monthly_window_count_x_mean``.
COUNT_X_MEAN_OPERAND_ROLES = ("count", "mean")
ALLOWED_VALUE_OPERATIONS = frozenset(
    (
        "identity",
        "sum",
        "difference",
        "linear_combination",
        "calendar_year_average",
        "calendar_year_window",
        "latest_plateau",
        "count_x_mean",
        "scaled_by_ratio",
        ROLLED_FORWARD_BY_RATIO,
        ALLOCATED_RESIDUAL_BY_RATIO,
        CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO,
        MONTHLY_WINDOW_COUNT_X_MEAN,
        *MONTHLY_WINDOW_OPERATIONS,
    )
)
MULTI_FACT_VALUE_OPERATIONS = frozenset(
    (
        "sum",
        "difference",
        "linear_combination",
        "calendar_year_average",
        "calendar_year_window",
        "latest_plateau",
        "count_x_mean",
        "scaled_by_ratio",
        ROLLED_FORWARD_BY_RATIO,
        ALLOCATED_RESIDUAL_BY_RATIO,
        CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO,
        MONTHLY_WINDOW_COUNT_X_MEAN,
        *MONTHLY_WINDOW_OPERATIONS,
    )
)
#: Operations that compose or pin periods of their own, so a reference may
#: not move its fact selection with ``observation_period``.
PERIOD_COMPOSING_OPERATIONS = frozenset(
    (
        "calendar_year_average",
        "calendar_year_window",
        "latest_plateau",
        "scaled_by_ratio",
        ROLLED_FORWARD_BY_RATIO,
        CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO,
    )
)
EXACT_PERIOD_VALUE_OPERATIONS = frozenset(
    (
        "identity",
        "sum",
        "difference",
        "linear_combination",
        "count_x_mean",
        "scaled_by_ratio",
        ALLOCATED_RESIDUAL_BY_RATIO,
    )
)

#: ``calendar_year_window``: a calendar-year target composed from the two
#: fiscal or tax years that overlap it, weighted by the months each
#: contributes (April-to-March years: three months of the year opening in
#: Y-1, nine of the year opening in Y). The weights are the operation's
#: definition, not a per-reference declaration, so a window row means one
#: thing everywhere it appears.
CALENDAR_YEAR_WINDOW_WEIGHTS: Mapping[int, float] = MappingProxyType(
    {-1: 3 / 12, 0: 9 / 12}
)
CALENDAR_YEAR_WINDOW_PERIOD_TYPES = frozenset(("fiscal_year", "tax_year"))
#: A ``calendar_year_window`` selector key for publishers who give each year
#: its own measure (DWP's benefit expenditure tables: ``expenditure_2024``,
#: ``expenditure_2025``). It maps each overlapping year, by its opening year,
#: to the measure that carries it: a fact matches only the measure its own
#: year declares, and the window reads the declared measures as one series.
SOURCE_MEASURE_ID_BY_OPENING_YEAR = "source_measure_id_by_opening_year"
#: The same for a series whose years sit in different editions of one
#: publication. OBR's EFO keeps one number per line and year, the latest
#: edition's (PolicyEngine/chronicle#397), so FY2024-25 sits in March 2026's
#: tables and FY2025-26 in October 2026's. The key maps each overlapping year,
#: by its opening year, to the source table that carries it: a fact matches
#: only the table its own year declares, and the window reads the declared
#: tables as one series.
SOURCE_TABLE_BY_OPENING_YEAR = "source_table_by_opening_year"
#: Ordered operand roles of ``scaled_by_ratio``: the published cell the
#: reference selects, then the two national facts whose quotient translates
#: it (a subset share of the publisher's universe, so the ratio lies in
#: (0, 1]). The base operand carries no selector of its own; the other two
#: are complete selectors, inheriting only the source name and period pins.
SCALED_BY_RATIO_OPERAND_ROLES = ("base", "numerator", "denominator")
#: Keys of a ``linear_combination`` operand that are not selector overlays.
LINEAR_COMBINATION_OPERAND_KEYS = frozenset(
    ("weight", "expected_member_count", "label", "dimension_values")
)
DEFAULT_HIERARCHY_MATCH_SPEC_FIELDS = ("entity", "period", "family", "filter")


class LedgerHierarchyMetadataError(ValueError):
    """A selected Chronicle fact cannot populate the required hierarchy."""


def hierarchy_seed_from_catalog(
    hierarchy: Mapping[str, object],
    category_id: str,
    *,
    target_id: str | None = None,
) -> CalibrationHierarchySeed:
    """Resolve one category reference from normalized provider/category catalogs."""

    providers = hierarchy.get("providers")
    categories = hierarchy.get("categories")
    if not isinstance(providers, Mapping) or not isinstance(categories, Mapping):
        raise ValueError("Hierarchy providers and categories must be mappings.")
    category = categories.get(category_id)
    if not isinstance(category, Mapping):
        raise ValueError(f"Unknown hierarchy category {category_id!r}.")
    provider_id = str(category.get("provider_id") or "")
    provider = providers.get(provider_id)
    if not isinstance(provider, Mapping):
        raise ValueError(
            f"Hierarchy category {category_id!r} references unknown provider "
            f"{provider_id!r}."
        )
    target_labels = hierarchy.get("target_labels")
    if target_labels is not None and not isinstance(target_labels, Mapping):
        raise ValueError("Hierarchy target_labels must be a mapping when present.")
    target_label = ""
    if isinstance(target_labels, Mapping) and target_id in target_labels:
        target_label = str(target_labels[target_id] or "").strip()
        if not target_label:
            raise ValueError(
                f"Hierarchy target {target_id!r} has an empty target label."
            )
    return CalibrationHierarchySeed(
        provider=HierarchyNode(
            id=provider_id,
            label=str(provider.get("label") or ""),
        ),
        category=HierarchyCategory(
            id=category_id,
            label=str(category.get("label") or ""),
            provider_id=provider_id,
        ),
        target_label=target_label or None,
    )


@dataclass(frozen=True)
class LedgerTargetMapping:
    """How Microcosm maps Ledger facts to model-ready target rows.

    A Ledger fact is source truth; it is not automatically a Microcosm target.
    Microcosm must know which model column estimates the fact's measure, and
    optionally which model boolean column implements the fact's scoped
    population. Facts without a mapping are reported as unsupported.
    """

    measure_by_concept: Mapping[str, str] = field(default_factory=dict)
    measure_by_source_record_id: Mapping[str, str] = field(default_factory=dict)
    filter_by_source_record_id: Mapping[str, str] = field(default_factory=dict)
    filter_by_domain: Mapping[str, str] = field(default_factory=dict)
    signed_by_concept: frozenset[str] = frozenset()
    signed_by_source_record_id: frozenset[str] = frozenset()
    entity_by_ledger_entity: Mapping[str, str] = field(default_factory=dict)
    family_by_source_name: Mapping[str, str] = field(default_factory=dict)
    family_by_concept: Mapping[str, str] = field(default_factory=dict)
    default_family: str = "ledger"


@dataclass(frozen=True)
class LedgerTargetReference:
    """A Microcosm calibration target backed by one Ledger fact.

    Ledger owns the observed fact: value, source lineage, source dimensions, and
    period. Microcosm owns whether that fact is active for a build and how it
    maps into the model: entity, measure column, optional filter, calibration
    family, and release-gate metadata. Uprating fields are declarations for the
    materialization layer; this compiler records them in target metadata after a
    caller has resolved the appropriate Ledger fact/value.
    """

    name: str
    ledger_fact_key: str = ""
    ledger_source_record_id: str = ""
    ledger_selector: Mapping[str, object] = field(default_factory=dict)
    value_operation: str = "identity"
    value_operands: tuple[Mapping[str, object], ...] = ()
    expected_member_count: int | None = None
    entity: str = ""
    measure: str | None = None
    filter: str | None = None
    period: int | str | None = None
    source: str | None = None
    family: str = "ledger"
    signed: bool = False
    se: float | None = None
    tolerance: float | None = None
    notes: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    hierarchy: CalibrationHierarchySeed | None = None
    assertion_policy: str = "observed_only"
    period_match_policy: str = "latest_not_after"
    #: The period the reference's facts are selected at, where it differs
    #: from the target period the model is measured at: DfE's January census
    #: for model year Y is January Y + 1, the spring term of the year's
    #: entitlements (microcosm#1095 D5). ``None`` selects at the target period.
    observation_period: int | str | None = None
    uprating_index: str | None = None
    uprating_from_period: int | str | None = None
    uprating_to_period: int | str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LedgerTargetReference.name must be non-empty.")
        if (
            not self.ledger_fact_key
            and not self.ledger_source_record_id
            and not self.ledger_selector
        ):
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: either ledger_fact_key "
                "or ledger_source_record_id or ledger_selector is required."
            )
        if self.value_operation not in ALLOWED_VALUE_OPERATIONS:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: unsupported value_operation "
                f"{self.value_operation!r}. Microcosm currently permits only "
                f"{sorted(ALLOWED_VALUE_OPERATIONS)!r} resolution from Ledger facts."
            )
        if not isinstance(self.ledger_selector, Mapping):
            raise TypeError(
                f"LedgerTargetReference {self.name!r}: ledger_selector must be a "
                f"mapping, got {type(self.ledger_selector).__name__}."
            )
        if self.expected_member_count is not None and (
            isinstance(self.expected_member_count, bool)
            or not isinstance(self.expected_member_count, int)
            or self.expected_member_count <= 0
        ):
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: expected_member_count "
                "must be a positive integer."
            )
        if self.value_operation == "difference":
            roles = [str(operand.get("role")) for operand in self.value_operands]
            if roles != ["minuend", "subtrahend"]:
                raise ValueError(
                    f"LedgerTargetReference {self.name!r}: difference requires "
                    "exactly ordered minuend/subtrahend operands."
                )
        if self.value_operation == "linear_combination":
            _validate_linear_combination_operands(self.name, self.value_operands)
        if self.value_operation == "scaled_by_ratio":
            _validate_scaled_by_ratio_operands(self.name, self.value_operands)
        if self.value_operation == ROLLED_FORWARD_BY_RATIO:
            _validate_rolled_forward_operands(self.name, self.value_operands)
            _rolled_forward_grouped_member_roster(self)
        if self.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
            _validate_allocation_reference(self)
        if self.value_operation == "calendar_year_window":
            _validate_calendar_year_window_reference(self)
        elif self.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO:
            _validate_calendar_year_window_scaled_by_ratio_reference(self)
        else:
            for key in (
                SOURCE_MEASURE_ID_BY_OPENING_YEAR,
                SOURCE_TABLE_BY_OPENING_YEAR,
            ):
                if key in self.ledger_selector:
                    raise ValueError(
                        f"LedgerTargetReference {self.name!r}: {key!r} composes "
                        "the years of a calendar_year_window; it is refused under "
                        f"value_operation {self.value_operation!r}."
                    )
        if self.value_operation == MONTHLY_WINDOW_COUNT_X_MEAN:
            _validate_count_x_mean_operands(self.name, self.value_operands)
        if self.assertion_policy not in ALLOWED_ASSERTION_POLICIES:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: unsupported "
                f"assertion_policy {self.assertion_policy!r}."
            )
        if self.period_match_policy not in ALLOWED_PERIOD_MATCH_POLICIES:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: unsupported "
                f"period_match_policy {self.period_match_policy!r}; expected "
                f"one of {sorted(ALLOWED_PERIOD_MATCH_POLICIES)!r}."
            )
        is_window = self.value_operation in {
            *MONTHLY_WINDOW_OPERATIONS,
            MONTHLY_WINDOW_COUNT_X_MEAN,
        }
        if self.observation_period is not None:
            if not _period_key_from_value(self.observation_period)[0]:
                raise ValueError(
                    f"LedgerTargetReference {self.name!r}: observation_period "
                    f"{self.observation_period!r} is not a parseable period."
                )
            if is_window or self.value_operation in PERIOD_COMPOSING_OPERATIONS:
                raise ValueError(
                    f"LedgerTargetReference {self.name!r}: observation_period "
                    "selects facts at one period, but value_operation "
                    f"{self.value_operation!r} composes periods of its own."
                )
        if is_window != (self.period_match_policy == "source_window"):
            raise ValueError(
                "A monthly window operation requires the paired source_window period policy."
            )
        if is_window:
            if self.period is None:
                raise ValueError(
                    "A monthly window requires an explicit model target period."
                )
            _declared_source_months(self)
            if self.value_operation in MONTHLY_WINDOW_OPERATIONS:
                _monthly_window_operands(self)
        if self.period_match_policy == "exact" and self.period is None:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: period_match_policy="
                "'exact' requires an explicit target period."
            )
        if (
            self.period_match_policy == "exact"
            and self.value_operation not in EXACT_PERIOD_VALUE_OPERATIONS
        ):
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: period_match_policy="
                f"'exact' does not support value_operation {self.value_operation!r}; "
                "calendar_year_average and latest_plateau consume subperiod "
                "series whose selection semantics are not an exact scalar-period "
                "match. Use latest_not_after or add an explicit subperiod "
                "materialization contract before activating this reference."
            )
        if not self.entity:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: entity must be non-empty."
            )
        if not self.family:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: family must be non-empty."
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"LedgerTargetReference {self.name!r}: metadata must be a "
                f"mapping, got {type(self.metadata).__name__}."
            )
        metadata = {str(key): str(value) for key, value in self.metadata.items()}
        bad_metadata = sorted(
            key for key, value in metadata.items() if not key or not value
        )
        if bad_metadata:
            raise ValueError(
                f"LedgerTargetReference {self.name!r}: metadata keys and values "
                f"must be non-empty strings; bad keys {bad_metadata}."
            )
        object.__setattr__(self, "metadata", metadata)
        if isinstance(self.hierarchy, Mapping):
            object.__setattr__(
                self,
                "hierarchy",
                CalibrationHierarchySeed.from_dict(dict(self.hierarchy)),
            )
        elif self.hierarchy is not None and not isinstance(
            self.hierarchy, CalibrationHierarchySeed
        ):
            raise TypeError(
                f"LedgerTargetReference {self.name!r}: hierarchy must be a "
                "CalibrationHierarchySeed, mapping, or None."
            )


@dataclass(frozen=True)
class LedgerTargetParityReport:
    """Exact and calibration-effective parity between two target registries."""

    passed: bool
    failures: tuple[str, ...]
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UnsupportedLedgerTarget:
    """A Ledger fact Microcosm deliberately did not activate."""

    reason: str
    identifier: str
    concept: str = ""
    source_record_id: str = ""


@dataclass(frozen=True)
class LedgerTargetSelection:
    """Selected Microcosm targets plus explicit unsupported Ledger facts."""

    specs: tuple[TargetSpec, ...]
    unsupported: tuple[UnsupportedLedgerTarget, ...]

    def to_registry(self, *, country: str) -> TargetRegistry:
        """Build a Microcosm target registry from the selected specs."""
        return TargetRegistry(self.specs, country=country)


@dataclass(frozen=True)
class LedgerTargetHierarchyRule:
    """Spec-declared parent/child reconciliation for Ledger-backed targets."""

    rule_id: str
    child_geography_level: str
    parent_geography_level: str
    parent_geography_id_template: str
    method: str = "scale_children_to_parent"
    match_spec_fields: tuple[str, ...] = DEFAULT_HIERARCHY_MATCH_SPEC_FIELDS
    match_metadata_keys: tuple[str, ...] = ()
    enabled_when: Mapping[str, str] = field(default_factory=dict)
    child_id_metadata_key: str = ""
    parent_key_metadata_key: str = ""
    expected_child_count_by_parent_key: Mapping[str, int] = field(default_factory=dict)
    on_incomplete: str = "fail"

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("LedgerTargetHierarchyRule.rule_id must be non-empty.")
        if self.method != "scale_children_to_parent":
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: unsupported "
                f"method {self.method!r}."
            )
        if not self.child_geography_level:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: "
                "child_geography_level is required."
            )
        if not self.parent_geography_level:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: "
                "parent_geography_level is required."
            )
        if not self.parent_geography_id_template:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: "
                "parent_geography_id.template is required."
            )
        object.__setattr__(
            self,
            "match_spec_fields",
            tuple(str(field_name) for field_name in self.match_spec_fields),
        )
        object.__setattr__(
            self,
            "match_metadata_keys",
            tuple(str(key) for key in self.match_metadata_keys),
        )
        object.__setattr__(
            self,
            "enabled_when",
            {
                str(key): _profile_scalar(value)
                for key, value in self.enabled_when.items()
            },
        )
        object.__setattr__(
            self,
            "expected_child_count_by_parent_key",
            {
                str(key): int(value)
                for key, value in self.expected_child_count_by_parent_key.items()
            },
        )
        if bool(self.expected_child_count_by_parent_key) != bool(
            self.child_id_metadata_key and self.parent_key_metadata_key
        ):
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: child "
                "completeness requires child_id_metadata_key, "
                "parent_key_metadata_key, and expected_child_count_by_parent_key."
            )
        bad_counts = sorted(
            key
            for key, value in self.expected_child_count_by_parent_key.items()
            if value <= 0
        )
        if bad_counts:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: expected child "
                f"counts must be positive; bad parent keys {bad_counts!r}."
            )
        if self.on_incomplete not in {"fail", "skip"}:
            raise ValueError(
                f"Ledger target hierarchy rule {self.rule_id!r}: unsupported "
                f"on_incomplete action {self.on_incomplete!r}."
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> LedgerTargetHierarchyRule:
        parent_geography_id = raw.get("parent_geography_id") or {}
        if not isinstance(parent_geography_id, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule parent_geography_id must be a mapping."
            )
        match = raw.get("match") or {}
        if not isinstance(match, Mapping):
            raise TypeError("Ledger target hierarchy rule match must be a mapping.")
        enabled_when = raw.get("enabled_when") or {}
        if not isinstance(enabled_when, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule enabled_when must be a mapping."
            )
        completeness = raw.get("child_completeness") or {}
        if not isinstance(completeness, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule child_completeness must be a mapping."
            )
        expected_child_counts = (
            completeness.get("expected_child_count_by_parent_key") or {}
        )
        if not isinstance(expected_child_counts, Mapping):
            raise TypeError(
                "Ledger target hierarchy rule "
                "expected_child_count_by_parent_key must be a mapping."
            )
        return cls(
            rule_id=str(raw.get("id") or ""),
            method=str(raw.get("method") or "scale_children_to_parent"),
            child_geography_level=str(raw.get("child_geography_level") or ""),
            parent_geography_level=str(raw.get("parent_geography_level") or ""),
            parent_geography_id_template=str(parent_geography_id.get("template") or ""),
            match_spec_fields=tuple(
                str(field_name)
                for field_name in (
                    match.get("spec_fields") or DEFAULT_HIERARCHY_MATCH_SPEC_FIELDS
                )
            ),
            match_metadata_keys=tuple(
                str(key) for key in (match.get("metadata_keys") or ())
            ),
            enabled_when=enabled_when,
            child_id_metadata_key=str(completeness.get("child_id_metadata_key") or ""),
            parent_key_metadata_key=str(
                completeness.get("parent_key_metadata_key") or ""
            ),
            expected_child_count_by_parent_key=expected_child_counts,
            on_incomplete=str(completeness.get("on_incomplete") or "fail"),
        )


@dataclass(frozen=True)
class _LedgerFactIndex:
    facts: tuple[object, ...]
    by_identifier: Mapping[str, tuple[object, ...]]
    by_aggregate_fact_key: Mapping[str, tuple[object, ...]]

    def lookup(self, identifier: str) -> tuple[object, ...]:
        return self.by_identifier.get(identifier, ())


def select_ledger_targets(
    facts: Iterable[object],
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> LedgerTargetSelection:
    """Select the Ledger facts Microcosm can target.

    Args:
        facts: Ledger ``AggregateFact`` objects or JSON-like mappings.
        mapping: Explicit Microcosm model-variable mapping.
        period: Optional period override for every resulting target.

    Returns:
        A selection with model-ready :class:`TargetSpec` rows and unsupported
        facts with reasons. Unsupported facts are a feature: the target catalog
        can be broader than Microcosm's current build surface.
    """

    specs: list[TargetSpec] = []
    unsupported: list[UnsupportedLedgerTarget] = []
    for fact in facts:
        try:
            specs.append(target_spec_from_ledger_fact(fact, mapping, period=period))
        except _UnsupportedFactError as exc:
            unsupported.append(exc.unsupported)
    return LedgerTargetSelection(tuple(specs), tuple(unsupported))


def select_ledger_targets_from_jsonl(
    path: str | Path,
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> LedgerTargetSelection:
    """Select Ledger facts from a JSON Lines consumer-contract file."""
    with Path(path).open() as file:
        return select_ledger_targets(
            _jsonl_rows(file),
            mapping,
            period=period,
        )


def compile_ledger_target_references(
    facts: Iterable[object],
    references: Iterable[LedgerTargetReference],
    *,
    country: str,
) -> TargetRegistry:
    """Resolve Microcosm target references against Ledger facts.

    This is the structured-registry path: each reference names the Ledger fact
    row it consumes, then overlays the model-facing calibration attributes that
    cannot live in the source database.
    """

    fact_index = _ledger_fact_index(facts)
    specs: list[TargetSpec] = []
    for reference in references:
        _require_executable_reference(reference)
        resolved = _resolve_reference_fact(reference, fact_index)
        specs.append(target_spec_from_ledger_reference(resolved, reference))
    return TargetRegistry(specs, country=country)


def _require_executable_reference(reference: LedgerTargetReference) -> None:
    """Refuse authoring placeholders until a scalar/fanout reference replaces them.

    Country packages may declare a future target surface before Chronicle has
    harvested and pinned the corresponding facts.  Those rows are useful schema
    evidence, but they are not executable selectors: in particular, resolving a
    multi-cell table through ``identity`` would either depend on accidental
    singleton input or become ambiguous as soon as the next cell arrived.
    Rows requiring a reference profile also carry the resolver's selection
    identity before an active status permits compilation.

    Those resolution markers are trusted declarations at this layer: this
    check sees that they are present, not that they match a resolution
    receipt. ``reference_profiles.validate_reference_resolution`` and the
    transport's effective-document binding authenticate a resolved document;
    callers compiling resolved references outside transport must run that
    validation first.
    """

    activation_status = reference.metadata.get("activation_status", "")
    if activation_status and activation_status != "active":
        raise ValueError(
            f"Ledger target reference {reference.name!r} is a non-executable "
            f"placeholder with activation_status={activation_status!r}. Replace "
            "it with harvested, cell-pinned references (or a reviewed scalar "
            "fact reference) before compilation."
        )

    if reference.metadata.get("required_reference_profile") and (
        reference.metadata.get("reference_profiles_resolved") != "1"
        or not reference.metadata.get("reference_selection_sha256")
    ):
        raise ValueError(
            f"Ledger target reference {reference.name!r} requires reference profile resolution before compilation."
        )


def apply_ledger_target_profile(
    registry: TargetRegistry,
    profile: Mapping[str, object] | None,
    *,
    context: Mapping[str, object] | None = None,
) -> TargetRegistry:
    """Apply spec-declared target-profile transformations.

    Ledger owns the raw facts. Microcosm target profiles can declare generic
    transformations needed to compile those facts into a coherent calibration
    surface, such as reconciling child-geography rows to an available parent.
    The declarations are data; this shared compiler owns the implementation.
    """

    if not profile:
        return registry
    schema_version = profile.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(
            "Ledger target profile schema_version must be an integer 1 or 2, "
            f"got {schema_version!r}."
        )
    if schema_version not in {1, 2}:
        raise ValueError(
            "Ledger target profile schema_version must be 1 or 2, got "
            f"{schema_version!r}."
        )
    result = registry
    for raw_rule in profile.get("hierarchy_reconciliations") or ():
        if not isinstance(raw_rule, Mapping):
            raise TypeError("Ledger hierarchy reconciliation entries must be mappings.")
        rule = LedgerTargetHierarchyRule.from_mapping(raw_rule)
        if _hierarchy_rule_enabled(rule, context or {}):
            result = reconcile_ledger_target_hierarchy(result, rule)
    return result


def reconcile_ledger_target_hierarchy(
    registry: TargetRegistry,
    rule: LedgerTargetHierarchyRule,
) -> TargetRegistry:
    """Scale child-geography target rows to their matching parent target row."""

    parent_by_key: dict[tuple[str, ...], TargetSpec] = {}
    child_groups: dict[tuple[str, ...], list[tuple[int, TargetSpec]]] = {}
    for index, spec in enumerate(registry.specs):
        geography_level = spec.metadata.get("ledger_geography_level", "")
        if geography_level == rule.parent_geography_level:
            key = _hierarchy_parent_key(spec, rule)
            existing = parent_by_key.get(key)
            if existing is not None:
                raise ValueError(
                    f"Ledger target hierarchy rule {rule.rule_id!r} found "
                    f"multiple parent targets for key {key!r}: "
                    f"{existing.name!r} and {spec.name!r}."
                )
            parent_by_key[key] = spec
        elif geography_level == rule.child_geography_level:
            parent_geography_id = _hierarchy_parent_geography_id(spec, rule)
            if not parent_geography_id:
                continue
            child_groups.setdefault(
                _hierarchy_match_key(spec, rule) + (parent_geography_id,),
                [],
            ).append((index, spec))

    replacements: dict[int, TargetSpec] = {}
    for key, children in child_groups.items():
        parent = parent_by_key.get(key)
        if parent is None:
            continue
        completeness = _hierarchy_completeness(parent, children, rule)
        if completeness is not None and not completeness["complete"]:
            if rule.on_incomplete == "skip":
                continue
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} expected "
                f"{completeness['expected_child_count']} child target(s) for "
                f"parent key {completeness['parent_key']!r} and parent "
                f"{parent.name!r}, got {completeness['observed_child_count']}."
            )
        raw_sum = sum(spec.value for _, spec in children)
        parent_value = parent.value
        if raw_sum == 0 and parent_value != 0:
            child_names = [spec.name for _, spec in children]
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} cannot scale "
                f"zero-valued children to nonzero parent {parent.name!r}; "
                f"children: {child_names!r}."
            )
        factor = 1.0 if raw_sum == 0 else parent_value / raw_sum
        if factor < 0:
            child_names = [spec.name for _, spec in children]
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} cannot reconcile "
                f"opposite-signed child and parent targets for {parent.name!r}; "
                f"children: {child_names!r}."
            )
        coverage_ratio = 1.0 if parent_value == 0 else raw_sum / parent_value
        for index, spec in children:
            replacements[index] = replace(
                spec,
                value=spec.value * factor,
                metadata={
                    **dict(spec.metadata),
                    "hierarchy_reconciliation_rule": rule.rule_id,
                    "hierarchy_reconciliation_method": rule.method,
                    "hierarchy_raw_value": _format_float(spec.value),
                    "hierarchy_child_sum_raw": _format_float(raw_sum),
                    "hierarchy_parent_value": _format_float(parent_value),
                    "hierarchy_coverage_ratio": _format_float(coverage_ratio),
                    "hierarchy_reconciliation_factor": _format_float(factor),
                    "hierarchy_parent_target_name": parent.name,
                    "hierarchy_parent_target_period": str(parent.period),
                    "hierarchy_parent_geography_level": rule.parent_geography_level,
                    "hierarchy_parent_geography_id": parent.metadata.get(
                        "ledger_geography_id", ""
                    ),
                    **_hierarchy_completeness_metadata(completeness),
                },
            )
    if not replacements:
        return registry
    specs = [replacements.get(index, spec) for index, spec in enumerate(registry.specs)]
    return TargetRegistry(specs, country=registry.country)


def target_spec_from_ledger_reference(
    fact: object | tuple[object, ...],
    reference: LedgerTargetReference,
) -> TargetSpec:
    """Compile one resolved Ledger fact plus Microcosm mapping into a target."""

    _require_executable_reference(reference)
    facts = fact if isinstance(fact, tuple) else (fact,)
    if not facts:
        raise ValueError(f"Ledger target reference {reference.name!r} has no facts.")
    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        facts = _resolve_monthly_window_reference_facts(reference, list(facts))
    elif reference.value_operation == MONTHLY_WINDOW_COUNT_X_MEAN:
        facts = _resolve_monthly_window_count_x_mean_reference_facts(
            reference, list(facts)
        )
    elif reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
        # Direct callers receive the same exact selector, universe and roster
        # checks as catalog callers; a pre-resolved tuple is never trusted.
        facts = _resolve_allocation_reference_facts(reference, facts)
    if len(facts) > 1 and reference.value_operation not in MULTI_FACT_VALUE_OPERATIONS:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: multiple Ledger facts "
            "require an aggregating value_operation, got "
            f"{reference.value_operation!r}."
        )
    numeric_values = []
    operand_facts = (
        sum(_window_ratio_operand_counts(reference))
        if reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO
        else 0
    )
    for index, member in enumerate(facts):
        if reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
            _validate_window_ratio_operand_fact(member, reference)
        elif (
            reference.value_operation == ROLLED_FORWARD_BY_RATIO
            and index >= len(facts) - 2
        ):
            # Growth operands may measure people while the base measures
            # households, and each pins its own geography vintage.
            selector = _rolled_forward_operand_selectors(reference)[
                index - len(facts) + 2
            ]
            if not _fact_matches_selector(member, selector):
                raise ValueError(
                    f"Ledger target reference {reference.name!r}: "
                    f"{ROLLED_FORWARD_BY_RATIO} resolved operand disagrees "
                    "with its declared selector."
                )
            _validate_window_ratio_operand_fact(member, reference)
        elif operand_facts and index >= len(facts) - operand_facts:
            # A ratio operand pins its own source, period and entity: the
            # reference's consumer pins describe the window, not the operand.
            _validate_window_ratio_operand_fact(member, reference)
        else:
            _validate_resolved_reference_fact(member, reference)
        if (
            reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO
            and "discrepancy_policy" in reference.value_operands[2]
        ):
            numeric_values.append(_exact_count_fact_value(member, reference))
        else:
            numeric_values.append(_numeric_fact_value(member, reference))
        _validate_fact_aggregation(member, reference)

    value_metadata: dict[str, str] = {}
    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        numeric_value = sum(numeric_values) / len(_declared_source_months(reference))
    elif reference.value_operation == "calendar_year_average":
        numeric_value = sum(numeric_values) / len(numeric_values)
    elif reference.value_operation == "latest_plateau":
        numeric_value = numeric_values[-1]
    elif reference.value_operation == "count_x_mean":
        numeric_value = numeric_values[0] * numeric_values[1]
    elif reference.value_operation == MONTHLY_WINDOW_COUNT_X_MEAN:
        numeric_value, value_metadata = _monthly_window_count_x_mean_value(
            reference, facts, numeric_values
        )
    elif reference.value_operation == "sum":
        numeric_value = sum(numeric_values)
    elif reference.value_operation == "difference":
        numeric_value = numeric_values[0] - numeric_values[1]
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: difference "
                f"produced invalid value {numeric_value!r}."
            )
    elif reference.value_operation == "scaled_by_ratio":
        numeric_value, value_metadata = _scaled_by_ratio_value(
            reference, numeric_values
        )
    elif reference.value_operation == ROLLED_FORWARD_BY_RATIO:
        numeric_value, value_metadata = _rolled_forward_by_ratio_value(
            reference, facts, numeric_values
        )
    elif reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
        numeric_value, value_metadata = _allocated_residual_value(reference, facts)
    elif reference.value_operation == "linear_combination":
        weights = _linear_combination_weights(reference, facts)
        numeric_value = sum(
            weight * value
            for weight, value in zip(weights, numeric_values, strict=True)
        )
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: linear_combination "
                f"produced invalid value {numeric_value!r}."
            )
    elif reference.value_operation == "calendar_year_window":
        weights = _calendar_year_window_weights(reference, facts)
        numeric_value = sum(
            weight * value
            for weight, value in zip(weights, numeric_values, strict=True)
        )
        value_metadata = _calendar_year_window_metadata(reference, facts)
    elif reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO:
        numerators, denominators = _window_ratio_operand_counts(reference)
        cut = len(facts) - numerators - denominators
        window = facts[:cut]
        weights = _calendar_year_window_weights(reference, window)
        base = math.fsum(
            weight * value
            for weight, value in zip(weights, numeric_values[:cut], strict=True)
        )
        numerator = math.fsum(numeric_values[cut : cut + numerators])
        denominator = math.fsum(numeric_values[cut + numerators :])
        complement = _window_ratio_numerator_is_complement(reference)
        if complement:
            numerator = denominator - numerator
        numeric_value, ratio_metadata = _scaled_by_ratio_value(
            reference, [base, numerator, denominator]
        )
        if complement:
            # The recorded numerator is the complement itself, so the formula
            # reads numerator / denominator (the review of microcosm#1422).
            ratio_metadata["ledger_value_numerator_rule"] = (
                "the denominator less the listed numerator cells"
            )
        value_metadata = {
            **_calendar_year_window_metadata(reference, window),
            **ratio_metadata,
        }
    else:
        numeric_value = numeric_values[0]
    representative_fact = _value_representative_fact(
        _window_ratio_window_facts(reference, facts)
        if reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO
        else facts,
        operation=reference.value_operation,
    )
    # A ratio-scaled cell keeps the base cell's identity: the national
    # numerator and denominator translate its value, they do not relocate it,
    # so the hierarchy and label read the base alone.
    identity_facts = (
        (representative_fact,)
        if reference.value_operation == "scaled_by_ratio"
        else tuple(
            member
            for member in facts
            if _fact_matches_selector(member, reference.ledger_selector)
        )
        if reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO
        else facts[:-2]
        if reference.value_operation == ROLLED_FORWARD_BY_RATIO
        else _window_ratio_window_facts(reference, facts)
        if reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO
        else facts
    )
    publication_metadata = dict(value_metadata)
    if reference.value_operation in {
        *MONTHLY_WINDOW_OPERATIONS,
        MONTHLY_WINDOW_COUNT_X_MEAN,
    }:
        # The window guard has proved these identities common to every member.
        publication_metadata = {
            **value_metadata,
            **{
                key: value
                for key, value in {
                    "ledger_source_release_key": _str_at(
                        representative_fact, "source_release_key"
                    ),
                    "ledger_source_sha256": _str_at(
                        representative_fact, "source", "source_sha256"
                    ),
                }.items()
                if value
            },
        }

    if not reference.measure:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: measure is required; "
            "count-like facts must be represented as sums of prepared indicator "
            "columns."
        )
    period = (
        reference.period
        if reference.period is not None
        else _at(representative_fact, "period", "value")
    )
    if period is None:
        raise ValueError(f"Ledger fact for {reference.name!r} is missing period.")

    return TargetSpec(
        name=reference.name,
        entity=reference.entity,
        measure=reference.measure,
        value=numeric_value,
        filter=reference.filter,
        period=period,
        se=reference.se,
        source=_source_citation(representative_fact),
        family=reference.family,
        signed=reference.signed,
        tolerance=reference.tolerance,
        notes=reference.notes,
        metadata={
            **_ledger_metadata(
                representative_fact,
                fact_key=_fact_key(representative_fact),
            ),
            **_multi_fact_reference_metadata(facts),
            **publication_metadata,
            **_diagnostic_target_label_metadata(identity_facts, target_period=period),
            "ledger_resolved_assertion": _fact_assertion(representative_fact),
            **_reference_metadata(reference),
        },
        hierarchy=_calibration_hierarchy(
            identity_facts,
            reference=reference,
            target_period=period,
        ),
    )


def _scaled_by_ratio_value(
    reference: LedgerTargetReference,
    numeric_values: list[float],
) -> tuple[float, dict[str, str]]:
    """Translate a published cell by the quotient of two facts.

    The result is ``base * numerator / denominator`` where the quotient must
    be a finite share in ``(0, 1]``: the operation exists to restate a cell
    published for a wider universe (all CGT taxpayers including trusts) in
    the narrower universe the model measures (individuals), never to
    inflate it. Every input and the ratio are recorded on the spec.
    """

    base, numerator, denominator = numeric_values
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: scaled_by_ratio "
            f"denominator {denominator!r} must be a finite positive value."
        )
    ratio = numerator / denominator
    if not math.isfinite(ratio) or ratio <= 0 or ratio > 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: scaled_by_ratio "
            f"produced ratio {ratio!r} from numerator {numerator!r} and "
            f"denominator {denominator!r}; a translated cell is a subset "
            "share of the published cell, so the ratio must lie in (0, 1]."
        )
    value = base * ratio
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: scaled_by_ratio "
            f"produced invalid value {value!r}."
        )
    return value, {
        "ledger_value_base": repr(float(base)),
        "ledger_value_numerator": repr(float(numerator)),
        "ledger_value_denominator": repr(float(denominator)),
        "ledger_value_ratio": repr(float(ratio)),
    }


def _rolled_forward_by_ratio_value(
    reference: LedgerTargetReference,
    facts: tuple[object, ...],
    numeric_values: list[float],
) -> tuple[float, dict[str, str]]:
    """Carry the base to the target period by the pinned operands' growth."""

    _assert_rolled_forward_base_cells(reference, list(facts[:-2]))
    *base_values, numerator, denominator = numeric_values
    base = math.fsum(base_values)
    ratio = _validated_growth_ratio(reference, facts[0], facts[-2], facts[-1])
    value = base * ratio
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: "
            f"{ROLLED_FORWARD_BY_RATIO} produced invalid value {value!r}."
        )
    geography: dict[str, str] = {}
    if _rolled_forward_base_overlay(reference):
        # The base sits below the reference's geography; the cell is the
        # reference's own area, so it keeps the pinned geography.
        geography = {
            key: str(reference.ledger_selector[key])
            for key in ("geography_level", "geography_id")
            if key in reference.ledger_selector
        }
    return value, {
        **geography,
        "ledger_value_base": repr(float(base)),
        "ledger_value_base_cells": str(len(base_values)),
        "ledger_value_base_fact_keys": json.dumps(
            [_fact_key(fact) or _source_record_id(fact) for fact in facts[:-2]],
            separators=(",", ":"),
        ),
        "ledger_value_numerator": repr(float(numerator)),
        "ledger_value_denominator": repr(float(denominator)),
        "ledger_value_ratio": repr(float(ratio)),
        "ledger_value_rolled_from_period": str(_comparable_period_value(facts[0])),
        "ledger_value_rolled_to_period": str(_comparable_period_value(facts[-2])),
    }


def _validated_growth_ratio(
    reference: LedgerTargetReference,
    base: object,
    numerator_fact: object,
    denominator_fact: object,
    *,
    exact: bool = False,
) -> float | Fraction:
    """Validate one series at the source and target periods, then its growth."""

    prefix = f"Ledger target reference {reference.name!r}: {reference.value_operation}"
    if _rolled_forward_series_key(numerator_fact) != _rolled_forward_series_key(
        denominator_fact
    ):
        raise ValueError(
            f"{prefix} numerator and denominator are not one series at two periods."
        )
    if _period_key(denominator_fact) != _period_key(base):
        raise ValueError(f"{prefix} denominator does not resolve at the base's period.")
    if _period_key(numerator_fact) != _period_key_from_value(reference.period):
        raise ValueError(f"{prefix} numerator does not resolve at the target period.")
    read_value = _exact_count_fact_value if exact else _numeric_fact_value
    numerator = read_value(numerator_fact, reference)
    denominator = read_value(denominator_fact, reference)
    if denominator <= 0 or (not exact and not math.isfinite(denominator)):
        raise ValueError(
            f"Ledger target reference {reference.name!r}: "
            f"{reference.value_operation} denominator {denominator!r} must be a "
            "finite positive value."
        )
    ratio = numerator / denominator
    low, high = ROLLED_FORWARD_RATIO_BOUNDS
    if exact:
        low, high = Fraction(str(low)), Fraction(str(high))
    if (not exact and not math.isfinite(ratio)) or not low < ratio <= high:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: "
            f"{reference.value_operation} produced ratio {ratio!r} from numerator "
            f"{numerator!r} and denominator {denominator!r}, outside "
            f"({low}, {high}]: a growth factor this large means an operand is "
            "pinned to the wrong period or series."
        )
    return ratio


def _value_representative_fact(
    facts: tuple[object, ...],
    *,
    operation: str,
) -> object:
    if len(facts) == 1 or operation not in MULTI_FACT_VALUE_OPERATIONS:
        return facts[0]
    if operation in {
        "scaled_by_ratio",
        ROLLED_FORWARD_BY_RATIO,
        ALLOCATED_RESIDUAL_BY_RATIO,
    }:
        return facts[0]
    return max(enumerate(facts), key=lambda item: (_period_key(item[1]), item[0]))[1]


def _numeric_fact_value(fact: object, reference: LedgerTargetReference) -> float:
    value = _at(fact, "value")
    if value is None:
        raise ValueError(f"Ledger fact for {reference.name!r} is missing value.")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Ledger fact for {reference.name!r} has invalid value {value!r}."
        ) from exc
    if not math.isfinite(numeric_value):
        raise ValueError(
            f"Ledger fact for {reference.name!r} has invalid value {value!r}."
        )
    return numeric_value


def _exact_count_fact_value(fact: object, reference: LedgerTargetReference) -> Fraction:
    """Read an integer count without introducing binary floating point arithmetic."""
    raw = _at(fact, "value")
    try:
        if isinstance(raw, bool) or raw is None:
            raise ValueError("not a count")
        value = Fraction(
            raw.lexical_token if isinstance(raw, LexicalFloat) else str(raw)
        )
        if value.denominator != 1 or value < 0:
            raise ValueError("not a nonnegative integer")
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"Ledger fact for {reference.name!r} requires an exact nonnegative integer count; got {raw!r}."
        ) from error
    return value


def _validate_fact_aggregation(
    fact: object,
    reference: LedgerTargetReference,
) -> None:
    aggregation = _str_at(fact, "aggregation", "method")
    accepts_time_mean = (
        reference.metadata.get("fact_aggregation") == "time_mean"
        and aggregation == "mean"
    )
    accepts_count_x_mean = aggregation == "mean" and (
        (
            reference.value_operation == "count_x_mean"
            and _count_mean_fact_role(fact) == "mean"
        )
        or (
            reference.value_operation == MONTHLY_WINDOW_COUNT_X_MEAN
            and _count_x_mean_operand_role(reference, fact) == "mean"
        )
    )
    if (
        aggregation not in SUPPORTED_LEDGER_AGGREGATIONS
        and not accepts_time_mean
        and not accepts_count_x_mean
    ):
        raise ValueError(
            f"Ledger fact for {reference.name!r} has unsupported aggregation "
            f"{aggregation!r}; Microcosm targets must be compiled from sum "
            "facts, including counts represented as sums of prepared indicator "
            "columns. A mean fact is only accepted when the Microcosm mapping "
            "declares fact_aggregation=time_mean, asserting the mean is taken "
            "over time periods on a stock count (still a linear level), never "
            "over entities (a per-unit ratio is not calibratable)."
        )


def _multi_fact_reference_metadata(facts: tuple[object, ...]) -> dict[str, str]:
    if len(facts) == 1:
        return {}
    member_keys = tuple(_fact_key(fact) or _source_record_id(fact) for fact in facts)
    payload = json.dumps(member_keys, separators=(",", ":"))
    varying_dimensions = _varying_dimension_ids(facts)
    return {
        "ledger_member_fact_count": str(len(facts)),
        "ledger_member_aggregate_fact_keys": payload,
        "ledger_member_fact_keys": payload,
        "ledger_member_fact_digest": hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
        **(
            {
                "ledger_aggregation_varying_dimensions": json.dumps(
                    varying_dimensions,
                    separators=(",", ":"),
                )
            }
            if varying_dimensions
            else {}
        ),
    }


def _calibration_hierarchy(
    facts: tuple[object, ...],
    *,
    reference: LedgerTargetReference,
    target_period: object,
) -> CalibrationHierarchy | None:
    """Complete the Microcosm hierarchy seed with Chronicle-owned fact data."""

    seed = reference.hierarchy
    if seed is None:
        return None
    _validate_chronicle_hierarchy_labels(facts, reference_name=reference.name)
    composed = _composed_row_geography(facts, reference=reference)
    if composed is None:
        composed = _aliased_row_geography(facts, reference=reference)
    if composed is not None:
        geography_level, geography_id, geography_label = composed
        return CalibrationHierarchy(
            provider=seed.provider,
            category=seed.category,
            geography=HierarchyGeography(
                id=geography_id,
                label=geography_label,
                level=geography_level,
            ),
            dimensions=_inherited_dimensions(facts),
            target=HierarchyNode(
                id=reference.name,
                label=_target_label(
                    facts, reference=reference, target_period=target_period
                ),
            ),
        )
    geography_pairs = {
        (
            _str_at(fact, "geography", "level"),
            _str_at(fact, "geography", "id"),
        )
        for fact in facts
    }
    if len(geography_pairs) != 1:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: hierarchy geography "
            f"must be constant across aggregate member facts, got "
            f"{sorted(geography_pairs)!r}."
        )
    geography_level, geography_id = next(iter(geography_pairs))
    if not geography_level or not geography_id:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: Chronicle geography "
            "level and id are required for the calibration hierarchy."
        )
    geography_names = {_str_at(fact, "geography", "name").strip() for fact in facts} - {
        ""
    }
    if len(geography_names) > 1:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: Chronicle geography "
            f"name conflicts across aggregate member facts: "
            f"{sorted(geography_names)!r}."
        )
    geography_label = (
        next(iter(geography_names))
        if geography_names
        else _geography_fallback_label(geography_id)
    )
    if not geography_label:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: Chronicle geography "
            f"{geography_level!r}/{geography_id!r} has no display label and is "
            "not present in Microcosm's authoritative geography catalog."
        )
    target_label = _target_label(
        facts,
        reference=reference,
        target_period=target_period,
    )
    return CalibrationHierarchy(
        provider=seed.provider,
        category=seed.category,
        geography=HierarchyGeography(
            id=geography_id,
            label=geography_label,
            level=geography_level,
        ),
        dimensions=_inherited_dimensions(facts),
        target=HierarchyNode(
            id=reference.name,
            label=target_label,
        ),
    )


def _target_label(
    facts: tuple[object, ...],
    *,
    reference: LedgerTargetReference,
    target_period: object,
) -> str:
    """Resolve the label from the owner of the resulting target semantics."""

    if _requires_microcosm_target_label(
        facts,
        reference=reference,
        target_period=target_period,
    ):
        if reference.hierarchy is None or not reference.hierarchy.target_label:
            raise LedgerHierarchyMetadataError(
                f"Ledger target reference {reference.name!r}: multi-fact, "
                "transformed, or restamped targets require an explicit "
                "Microcosm target label."
            )
        return reference.hierarchy.target_label

    fact = facts[0]
    label = _str_at(fact, "label").strip()
    if not label:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: the selected Chronicle "
            "fact requires a non-empty label."
        )
    return label


def _validate_chronicle_hierarchy_labels(
    facts: tuple[object, ...],
    *,
    reference_name: str,
) -> None:
    """Require Chronicle-owned labels without identifier-derived substitutes."""

    dimension_labels: dict[str, set[str]] = {}
    dimension_value_labels: dict[tuple[str, str], set[str]] = {}
    for index, fact in enumerate(facts):
        fact_id = _fact_key(fact) or _source_record_id(fact) or f"member[{index}]"
        if not _str_at(fact, "label").strip():
            raise LedgerHierarchyMetadataError(
                f"Ledger target reference {reference_name!r}: every selected "
                f"Chronicle fact requires a non-empty label; missing for {fact_id!r}."
            )

        raw_dimensions = _dimensions(fact)
        dimensions = {str(key): value for key, value in raw_dimensions.items()}
        groupby_id = _str_at(fact, "layout", "groupby_dimension").strip()
        groupby_value_id = _str_at(fact, "layout", "groupby_value_id").strip()
        if groupby_id and groupby_value_id:
            dimensions.setdefault(groupby_id, groupby_value_id)

        for dimension_id, value in dimensions.items():
            labels = {
                str(label).strip()
                for label in (
                    _mapping_value(fact, "dimension_labels", dimension_id),
                    (
                        _at(fact, "layout", "groupby_dimension_label")
                        if dimension_id == groupby_id
                        else None
                    ),
                )
                if label is not None and str(label).strip()
            }
            if len(labels) != 1:
                raise LedgerHierarchyMetadataError(
                    f"Ledger target reference {reference_name!r}: Chronicle fact "
                    f"{fact_id!r} dimension {dimension_id!r} requires exactly one "
                    f"non-empty label, got {sorted(labels)!r}."
                )
            dimension_labels.setdefault(dimension_id, set()).update(labels)

            value_id = _dimension_value_id(value)
            value_labels = {
                str(label).strip()
                for label in (
                    _nested_mapping_value(
                        fact,
                        "dimension_value_labels",
                        dimension_id,
                        value_id,
                    ),
                    (
                        _at(fact, "layout", "groupby_value_label")
                        if dimension_id == groupby_id and value_id == groupby_value_id
                        else None
                    ),
                )
                if label is not None and str(label).strip()
            }
            if len(value_labels) != 1:
                raise LedgerHierarchyMetadataError(
                    f"Ledger target reference {reference_name!r}: Chronicle fact "
                    f"{fact_id!r} dimension {dimension_id!r} value {value_id!r} "
                    "requires exactly one non-empty label, got "
                    f"{sorted(value_labels)!r}."
                )
            dimension_value_labels.setdefault((dimension_id, value_id), set()).update(
                value_labels
            )

    for dimension_id, labels in dimension_labels.items():
        if len(labels) != 1:
            raise LedgerHierarchyMetadataError(
                f"Ledger target reference {reference_name!r}: Chronicle dimension "
                f"{dimension_id!r} has conflicting labels {sorted(labels)!r}."
            )
    for (dimension_id, value_id), labels in dimension_value_labels.items():
        if len(labels) != 1:
            raise LedgerHierarchyMetadataError(
                f"Ledger target reference {reference_name!r}: Chronicle dimension "
                f"{dimension_id!r} value {value_id!r} has conflicting labels "
                f"{sorted(labels)!r}."
            )


def _requires_microcosm_target_label(
    facts: tuple[object, ...],
    *,
    reference: LedgerTargetReference,
    target_period: object,
) -> bool:
    if len(facts) != 1 or reference.value_operation != "identity":
        return True
    fact = facts[0]
    if _fact_key(fact).startswith("microcosm.derived_fact."):
        return True
    fact_period = _at(fact, "period", "value")
    if fact_period is None:
        return True
    return not period_values_semantically_equal(
        target_period,
        fact_period,
        declared_type=_str_at(fact, "period", "type"),
    )


def _inherited_dimensions(facts: tuple[object, ...]) -> tuple[HierarchyDimension, ...]:
    """Return dimensions constant across all facts in deterministic order."""

    common = _common_dimensions(facts)
    dimensions: list[HierarchyDimension] = []
    groupby_id = _constant_text(facts, "layout", "groupby_dimension")
    groupby_value_id = _constant_text(facts, "layout", "groupby_value_id")
    if groupby_id and groupby_value_id:
        raw_value = common.pop(groupby_id, groupby_value_id)
        dimensions.append(
            HierarchyDimension(
                id=groupby_id,
                label=(
                    _constant_text(facts, "layout", "groupby_dimension_label")
                    or _dimension_label(facts, groupby_id)
                ),
                value_id=_dimension_value_id(groupby_value_id),
                value_label=(
                    _constant_text(facts, "layout", "groupby_value_label")
                    or _dimension_value_label(facts, groupby_id, raw_value)
                ),
            )
        )
    for dimension_id in sorted(common):
        value = common[dimension_id]
        dimensions.append(
            HierarchyDimension(
                id=dimension_id,
                label=_dimension_label(facts, dimension_id),
                value_id=_dimension_value_id(value),
                value_label=_dimension_value_label(facts, dimension_id, value),
            )
        )
    return tuple(dimensions)


def _common_dimensions(facts: tuple[object, ...]) -> dict[str, object]:
    first = {str(key): value for key, value in _dimensions(facts[0]).items()}
    return {
        key: value
        for key, value in first.items()
        if all(
            key in _dimensions(fact) and _dimensions(fact)[key] == value
            for fact in facts[1:]
        )
    }


def _varying_dimension_ids(facts: tuple[object, ...]) -> list[str]:
    if len(facts) < 2:
        return []
    dimension_ids = sorted(
        {str(key) for fact in facts for key in _dimensions(fact).keys()}
    )
    return [
        dimension_id
        for dimension_id in dimension_ids
        if len(
            {
                json.dumps(
                    _dimensions(fact).get(dimension_id),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for fact in facts
            }
        )
        > 1
    ]


def _constant_text(facts: tuple[object, ...], *path: str) -> str:
    values = {_str_at(fact, *path).strip() for fact in facts}
    return next(iter(values)) if len(values) == 1 else ""


def _dimension_label(facts: tuple[object, ...], dimension_id: str) -> str:
    labels = {
        str(label).strip()
        for fact in facts
        for label in (_mapping_value(fact, "dimension_labels", dimension_id),)
        if label is not None and str(label).strip()
    }
    if len(labels) != 1:
        raise LedgerHierarchyMetadataError(
            f"Chronicle dimension {dimension_id!r} requires exactly one "
            f"non-empty label across selected facts, got {sorted(labels)!r}."
        )
    return next(iter(labels))


def _dimension_value_label(
    facts: tuple[object, ...], dimension_id: str, value: object
) -> str:
    labels = {
        str(label).strip()
        for fact in facts
        for label in (
            _nested_mapping_value(
                fact,
                "dimension_value_labels",
                dimension_id,
                _dimension_value_id(value),
            ),
        )
        if label is not None and str(label).strip()
    }
    if len(labels) != 1:
        value_id = _dimension_value_id(value)
        raise LedgerHierarchyMetadataError(
            f"Chronicle dimension {dimension_id!r} value {value_id!r} requires "
            "exactly one non-empty label across selected facts, got "
            f"{sorted(labels)!r}."
        )
    return next(iter(labels))


def _mapping_value(fact: object, mapping_name: str, key: str) -> object | None:
    mapping = _at(fact, mapping_name)
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def _nested_mapping_value(
    fact: object, mapping_name: str, outer_key: str, inner_key: str
) -> object | None:
    outer = _mapping_value(fact, mapping_name, outer_key)
    return outer.get(inner_key) if isinstance(outer, Mapping) else None


def _dimension_value_id(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _composed_row_geography(
    facts: tuple[object, ...],
    *,
    reference: LedgerTargetReference,
) -> tuple[str, str, str] | None:
    """The declared geography of a row composed from lower-grain member facts.

    A reference whose metadata carries ``composed_from_level`` sums (or
    signs) member facts that sit one grain below the row it declares in its
    own ``geography_level``/``geography_id`` (a region row built from its
    authorities' rows, microcosm#929). Chronicle names each member area, not
    the composed one, so the row's label comes from Microcosm's authoritative
    geography catalog; a code the catalog does not carry refuses, the same
    way an unnamed sub-national fact does.
    """

    composed_from = str(reference.metadata.get("composed_from_level") or "").strip()
    if not composed_from:
        return None
    level = str(reference.metadata.get("geography_level") or "").strip()
    geography_id = str(reference.metadata.get("geography_id") or "").strip()
    if not level or not geography_id:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: a composed row must "
            "declare its own geography_level and geography_id."
        )
    member_levels = {_str_at(fact, "geography", "level") for fact in facts}
    if member_levels != {composed_from}:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: composed row members "
            f"must all sit at {composed_from!r}, got {sorted(member_levels)!r}."
        )
    label = _geography_fallback_label(geography_id)
    if not label:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: composed geography "
            f"{level!r}/{geography_id!r} is not present in Microcosm's "
            "authoritative geography catalog."
        )
    return level, geography_id, label


def _aliased_row_geography(
    facts: tuple[object, ...],
    *,
    reference: LedgerTargetReference,
) -> tuple[str, str, str] | None:
    """The roster geography of a row whose facts carry a publisher's alias code.

    A reference whose metadata lists ``geography_id_aliases`` may resolve facts
    stamped with an alias code (an authority the publisher recoded after the
    roster's boundary vintage); the row keeps the roster code as its identity
    and the publisher's name for the area as its label.
    """

    aliases = {
        code.strip()
        for code in str(reference.metadata.get("geography_id_aliases") or "").split(",")
        if code.strip()
    }
    if not aliases:
        return None
    fact_ids = {_str_at(fact, "geography", "id") for fact in facts}
    if not fact_ids & aliases:
        return None
    level = str(reference.metadata.get("geography_level") or "").strip()
    geography_id = str(reference.metadata.get("geography_id") or "").strip()
    if not level or not geography_id:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: an aliased row must "
            "declare its own geography_level and geography_id."
        )
    if fact_ids - aliases - {geography_id}:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: aliased row members "
            f"must sit at the roster code or a declared alias, got "
            f"{sorted(fact_ids)!r}."
        )
    names = {_str_at(fact, "geography", "name").strip() for fact in facts} - {""}
    if len(names) != 1:
        raise LedgerHierarchyMetadataError(
            f"Ledger target reference {reference.name!r}: aliased row needs one "
            f"publisher name for the area, got {sorted(names)!r}."
        )
    return level, geography_id, next(iter(names))


def _geography_fallback_label(geography_id: str) -> str:
    return UK_GEOGRAPHY_ID_TO_LABEL.get(geography_id, "")


def ledger_target_registry_parity_report(
    expected: TargetRegistry,
    actual: TargetRegistry,
) -> LedgerTargetParityReport:
    """Compare target registries before switching an active calibration path.

    The exact digest covers full :class:`TargetSpec` records. The calibration
    digest covers the fields consumed by solver compilation, so lineage-only
    registry additions can be reviewed without disguising calibration changes.
    """

    expected_exact = _registry_digest(expected)
    actual_exact = _registry_digest(actual)
    expected_calibration = _registry_digest(
        expected, payload_fn=_calibration_effective_spec_payload
    )
    actual_calibration = _registry_digest(
        actual, payload_fn=_calibration_effective_spec_payload
    )

    failures: list[str] = []
    if expected.country != actual.country:
        failures.append(
            f"country differs: expected {expected.country!r}, got {actual.country!r}."
        )
    if len(expected) != len(actual):
        failures.append(
            f"target count differs: expected {len(expected)}, got {len(actual)}."
        )
    if expected_exact != actual_exact:
        failures.append(
            f"full target registry digest differs: expected {expected_exact}, "
            f"got {actual_exact}."
        )
    if expected_calibration != actual_calibration:
        failures.append(
            "calibration-effective target digest differs: expected "
            f"{expected_calibration}, got {actual_calibration}."
        )
    if expected_exact != actual_exact or expected_calibration != actual_calibration:
        failures.extend(_first_spec_differences(expected, actual))

    return LedgerTargetParityReport(
        passed=not failures,
        failures=tuple(failures),
        details={
            "expected_count": len(expected),
            "actual_count": len(actual),
            "expected_version": expected.version,
            "actual_version": actual.version,
            "expected_full_digest": expected_exact,
            "actual_full_digest": actual_exact,
            "expected_calibration_digest": expected_calibration,
            "actual_calibration_digest": actual_calibration,
        },
    )


def target_spec_from_ledger_fact(
    fact: object,
    mapping: LedgerTargetMapping,
    *,
    period: int | str | None = None,
) -> TargetSpec:
    """Convert one supported Ledger fact to a Microcosm target spec.

    The target's ``name`` is the Ledger stable fact identifier when available,
    otherwise its source-record ID. Human-facing tooling should display the
    structured metadata fields, not this internal key.
    """

    source_record_id = _source_record_id(fact)
    fact_key = _fact_key(fact)
    identifier = fact_key or source_record_id
    if not identifier:
        raise _unsupported("missing_identifier", fact)

    concept = _primary_measure_concept(fact)
    if not concept:
        raise _unsupported("missing_measure_concept", fact)

    aggregation = _str_at(fact, "aggregation", "method")
    if aggregation not in SUPPORTED_LEDGER_AGGREGATIONS:
        raise _unsupported(f"unsupported_aggregation:{aggregation}", fact)

    measure = mapping.measure_by_source_record_id.get(source_record_id)
    if not measure:
        for candidate in _measure_concepts(fact):
            measure = mapping.measure_by_concept.get(candidate)
            if measure:
                break
    if not measure:
        raise _unsupported("missing_model_measure_mapping", fact)

    ledger_entity = _str_at(fact, "entity", "name")
    entity = mapping.entity_by_ledger_entity.get(ledger_entity, ledger_entity)
    if not entity:
        raise _unsupported("missing_entity", fact)

    value = _at(fact, "value")
    if value is None:
        raise _unsupported("missing_value", fact)
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _unsupported("invalid_value", fact) from exc
    if not math.isfinite(numeric_value):
        raise _unsupported("invalid_value", fact)

    target_period = period if period is not None else _at(fact, "period", "value")
    if target_period is None:
        raise _unsupported("missing_period", fact)

    source_name = _source_name(fact)
    family = (
        mapping.family_by_concept.get(concept)
        or mapping.family_by_source_name.get(source_name)
        or source_name
        or mapping.default_family
    )

    filter_column = mapping.filter_by_source_record_id.get(source_record_id)
    if filter_column is None and _requires_detail_filter_mapping(fact):
        raise _unsupported("missing_model_filter_mapping", fact)
    domain = _domain(fact)
    if filter_column is None and _requires_domain_filter_mapping(fact):
        filter_column = mapping.filter_by_domain.get(domain)
    if filter_column is None and _requires_filter_mapping(fact):
        raise _unsupported("missing_model_filter_mapping", fact)
    if numeric_value < 0 and not _allows_signed_target(fact, mapping):
        raise _unsupported("missing_signed_target_mapping", fact)

    try:
        return TargetSpec(
            name=identifier,
            entity=entity,
            measure=measure,
            value=numeric_value,
            filter=filter_column,
            period=target_period,
            source=_source_citation(fact),
            family=family,
            signed=numeric_value < 0,
            metadata={
                **_ledger_metadata(fact, fact_key=fact_key),
                **_diagnostic_target_label_metadata(
                    (fact,), target_period=target_period
                ),
            },
        )
    except (TypeError, ValueError) as exc:
        raise _unsupported(f"invalid_target_spec:{type(exc).__name__}", fact) from exc


class _UnsupportedFactError(Exception):
    def __init__(self, unsupported: UnsupportedLedgerTarget) -> None:
        super().__init__(unsupported.reason)
        self.unsupported = unsupported


def _unsupported(reason: str, fact: object) -> _UnsupportedFactError:
    return _UnsupportedFactError(
        UnsupportedLedgerTarget(
            reason=reason,
            identifier=_fact_key(fact) or _source_record_id(fact),
            concept=_primary_measure_concept(fact),
            source_record_id=_source_record_id(fact),
        )
    )


def _ledger_fact_index(facts: Iterable[object]) -> _LedgerFactIndex:
    materialized_facts = tuple(facts)
    buckets: dict[str, list[object]] = {}
    aggregate_buckets: dict[str, list[object]] = {}
    for fact in materialized_facts:
        aggregate_key = _str_at(fact, "aggregate_fact_key")
        if aggregate_key:
            # Keep every source occurrence, including repeated Python objects.
            aggregate_buckets.setdefault(aggregate_key, []).append(fact)
        keys = (
            aggregate_key,
            _str_at(fact, "semantic_fact_key"),
            _str_at(fact, "fact_key"),
            _str_at(fact, "legacy_fact_key"),
            _source_record_id(fact),
        )
        for key in dict.fromkeys(candidate for candidate in keys if candidate):
            bucket = buckets.setdefault(key, [])
            if not any(existing is fact for existing in bucket):
                bucket.append(fact)
    return _LedgerFactIndex(
        facts=materialized_facts,
        by_identifier={key: tuple(value) for key, value in buckets.items()},
        by_aggregate_fact_key={
            key: tuple(value) for key, value in aggregate_buckets.items()
        },
    )


def _resolve_reference_fact(
    reference: LedgerTargetReference,
    fact_index: _LedgerFactIndex,
) -> object:
    identifiers = tuple(
        key
        for key in (reference.ledger_fact_key, reference.ledger_source_record_id)
        if key
    )
    matches_by_identifier = {
        key: fact_index.lookup(key) for key in identifiers if fact_index.lookup(key)
    }
    ambiguous = {
        key: matches
        for key, matches in matches_by_identifier.items()
        if len(matches) > 1
    }
    if ambiguous:
        examples = {
            key: [
                _fact_key(fact) or _source_record_id(fact) or f"fact[{index}]"
                for index, fact in enumerate(matches[:5])
            ]
            for key, matches in ambiguous.items()
        }
        raise ValueError(
            f"Ledger target reference {reference.name!r} matched multiple "
            f"Ledger facts for identifier(s): {examples!r}."
        )
    resolved = {key: matches[0] for key, matches in matches_by_identifier.items()}
    missing = tuple(key for key in identifiers if key not in resolved)
    if resolved and missing:
        raise ValueError(
            f"Ledger target reference {reference.name!r} matched only some Ledger "
            f"fact identifiers. Missing: {missing!r}."
        )
    if resolved:
        distinct_facts = {id(fact) for fact in resolved.values()}
        if len(distinct_facts) > 1:
            raise ValueError(
                f"Ledger target reference {reference.name!r} identifiers resolve "
                "to different Ledger facts."
            )
        return next(iter(resolved.values()))
    if reference.ledger_selector:
        matches = [
            fact
            for fact in fact_index.facts
            if _fact_matches_selector(fact, reference.ledger_selector)
        ]
        eligible_matches = _eligible_selector_matches(reference, matches)
        if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
            return _resolve_monthly_window_reference_facts(reference, eligible_matches)
        if reference.value_operation == MONTHLY_WINDOW_COUNT_X_MEAN:
            return _resolve_monthly_window_count_x_mean_reference_facts(
                reference, eligible_matches
            )
        if reference.value_operation == "sum" and eligible_matches:
            return _resolve_sum_reference_facts(reference, eligible_matches)
        if reference.value_operation == "difference" and eligible_matches:
            return _resolve_difference_reference_facts(reference, eligible_matches)
        if reference.value_operation == "scaled_by_ratio" and eligible_matches:
            return _resolve_scaled_by_ratio_reference_facts(
                reference, eligible_matches, fact_index.facts
            )
        if reference.value_operation == ROLLED_FORWARD_BY_RATIO:
            return _resolve_rolled_forward_reference_facts(
                reference, eligible_matches, fact_index.facts
            )
        if reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
            return _resolve_allocation_reference_facts(
                reference,
                fact_index.facts,
                fact_index=fact_index,
                base_matches=matches,
            )
        if reference.value_operation == "linear_combination" and eligible_matches:
            return _resolve_linear_combination_reference_facts(
                reference, eligible_matches
            )
        if reference.value_operation == "calendar_year_average" and eligible_matches:
            return _resolve_calendar_year_average_reference_facts(
                reference, eligible_matches
            )
        if reference.value_operation == "calendar_year_window" and eligible_matches:
            return _resolve_calendar_year_window_reference_facts(
                reference, eligible_matches
            )
        if (
            reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO
            and eligible_matches
        ):
            return _resolve_calendar_year_window_scaled_by_ratio_reference_facts(
                reference, eligible_matches, fact_index.facts
            )
        if reference.value_operation == "latest_plateau" and eligible_matches:
            return _resolve_latest_plateau_reference_facts(reference, eligible_matches)
        if reference.value_operation == "count_x_mean" and eligible_matches:
            return _resolve_count_x_mean_reference_facts(reference, eligible_matches)
        if len(eligible_matches) == 1:
            return eligible_matches[0]
        latest_match = (
            None
            if reference.period_match_policy == "exact"
            else _latest_period_selector_match(reference, eligible_matches)
        )
        if latest_match is not None:
            return latest_match
        if not matches:
            raise ValueError(
                f"Ledger target reference {reference.name!r} did not match a "
                f"Ledger fact selector: {dict(reference.ledger_selector)!r}."
            )
        if not eligible_matches:
            period_requirement = (
                "at exact target period"
                if reference.period_match_policy == "exact"
                else "at or before target period"
            )
            raise ValueError(
                f"Ledger target reference {reference.name!r} did not match a "
                f"Ledger fact {period_requirement} "
                f"{_selection_period(reference)!r} for selector: "
                f"{dict(reference.ledger_selector)!r}."
            )
        identifiers = [
            _fact_key(fact) or _source_record_id(fact) or f"fact[{index}]"
            for index, fact in enumerate(eligible_matches)
        ]
        raise ValueError(
            f"Ledger target reference {reference.name!r} matched multiple "
            f"Ledger facts for selector {dict(reference.ledger_selector)!r}: "
            f"{identifiers!r}."
        )
    raise ValueError(
        f"Ledger target reference {reference.name!r} did not match a Ledger fact "
        f"identifier: {identifiers!r}."
    )


def _resolve_sum_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    partitions: dict[tuple[tuple[str, ...], tuple[int, int, str]], list[object]] = {}
    for fact in eligible_matches:
        key = (
            _selector_sum_partition_key(fact),
            _reference_period_partition_key(fact, reference),
        )
        partitions.setdefault(key, []).append(fact)
    if not partitions:
        raise ValueError(
            f"Ledger target reference {reference.name!r} did not match a Ledger "
            "fact partition."
        )
    latest_period = max(period_key for _, period_key in partitions)
    latest_matches = [
        fact
        for (_, period_key), facts in partitions.items()
        if period_key == latest_period
        for fact in facts
    ]
    if (
        reference.expected_member_count is not None
        and len(latest_matches) != reference.expected_member_count
    ):
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation=sum "
            f"expected {reference.expected_member_count} members at the latest "
            f"period but resolved {len(latest_matches)}; a declared member is missing."
        )
    return tuple(
        sorted(
            latest_matches,
            key=lambda fact: _fact_key(fact) or _source_record_id(fact),
        )
    )


def _resolve_difference_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    resolved: list[object] = []
    for operand in reference.value_operands:
        dimensions = operand.get("dimension_values")
        if not isinstance(dimensions, Mapping):
            raise ValueError(
                f"Ledger target reference {reference.name!r}: difference operand "
                "requires dimension_values."
            )
        shared_dimensions = reference.ledger_selector.get("dimension_values")
        selector = {
            **dict(reference.ledger_selector),
            **{
                str(key): value
                for key, value in operand.items()
                if key not in {"role", "dimension_values"}
            },
            "dimension_values": {
                **(
                    dict(shared_dimensions)
                    if isinstance(shared_dimensions, Mapping)
                    else {}
                ),
                **dict(dimensions),
            },
        }
        matches = [
            fact for fact in eligible_matches if _fact_matches_selector(fact, selector)
        ]
        match = _latest_period_selector_match(reference, matches)
        if match is None:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: difference operand "
                f"{operand.get('role')!r} did not resolve exactly once."
            )
        resolved.append(match)
    periods = {_period_key(fact) for fact in resolved}
    if len(periods) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: difference operands "
            "must resolve at the same latest period."
        )
    return tuple(resolved)


#: Base-selector pins a ``scaled_by_ratio`` operand inherits unless it
#: overrides them: the publisher and the observation period, so the quotient
#: is taken from the same release and year as the cell it translates.
_SCALED_BY_RATIO_INHERITED_SELECTOR_KEYS = (
    "source_name",
    "period_type",
    "period_value",
    "assertion",
)


def _validate_scaled_by_ratio_operands(
    name: str, operands: tuple[Mapping[str, object], ...]
) -> None:
    roles = [
        str(operand.get("role")) if isinstance(operand, Mapping) else ""
        for operand in operands
    ]
    if roles != list(SCALED_BY_RATIO_OPERAND_ROLES):
        raise ValueError(
            f"LedgerTargetReference {name!r}: scaled_by_ratio requires exactly "
            f"ordered {'/'.join(SCALED_BY_RATIO_OPERAND_ROLES)} operands, got "
            f"{roles!r}."
        )
    base_overrides = {key for key in operands[0] if key != "role"}
    if base_overrides:
        raise ValueError(
            f"LedgerTargetReference {name!r}: the scaled_by_ratio base operand "
            "is the reference's own selector and takes no overrides, got "
            f"{sorted(base_overrides)!r}."
        )
    for operand in operands[1:]:
        selector = {key for key in operand if key != "role"}
        if not selector:
            raise ValueError(
                f"LedgerTargetReference {name!r}: scaled_by_ratio operand "
                f"{operand.get('role')!r} needs at least one selector field."
            )


def _validate_rolled_forward_operands(
    name: str, operands: tuple[Mapping[str, object], ...]
) -> None:
    roles = [
        str(operand.get("role")) if isinstance(operand, Mapping) else ""
        for operand in operands
    ]
    prefix = f"LedgerTargetReference {name!r}: {ROLLED_FORWARD_BY_RATIO}"
    if roles != list(ROLLED_FORWARD_OPERAND_ROLES):
        raise ValueError(
            f"{prefix} requires exactly ordered "
            f"{'/'.join(ROLLED_FORWARD_OPERAND_ROLES)} operands, got {roles!r}."
        )
    count = operands[0].get("expected_member_count")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
    ):
        raise ValueError(
            f"{prefix}: base expected_member_count must be a positive integer."
        )
    base_overlay = {
        key for key in operands[0] if key not in {"role", "expected_member_count"}
    }
    if base_overlay - {"geography_level", "geography_id"}:
        raise ValueError(
            f"{prefix}: the base operand is the reference's own selector; it may "
            "only move the base below the reference's geography "
            f"(geography_level, geography_id), got {sorted(base_overlay)!r}."
        )
    for operand in operands[1:]:
        if "expected_member_count" in operand:
            raise ValueError(
                f"{prefix}: expected_member_count applies to base facts only."
            )
        if "period_type" not in operand or "period_value" not in operand:
            raise ValueError(
                f"{prefix}: operand {operand.get('role')!r} must pin its own "
                "period_type and period_value; a roll-forward reads each "
                "operand at a declared period, never at the latest one."
            )
        if not {key for key in operand} - {"role", "period_type", "period_value"}:
            raise ValueError(
                f"{prefix}: operand {operand.get('role')!r} needs at least one "
                "selector field besides its period."
            )


def _validate_count_x_mean_operands(
    name: str, operands: tuple[Mapping[str, object], ...]
) -> None:
    prefix = f"LedgerTargetReference {name!r}: {MONTHLY_WINDOW_COUNT_X_MEAN}"
    roles = [
        str(operand.get("role")) if isinstance(operand, Mapping) else ""
        for operand in operands
    ]
    if roles != list(COUNT_X_MEAN_OPERAND_ROLES):
        raise ValueError(
            f"{prefix} requires exactly ordered "
            f"{'/'.join(COUNT_X_MEAN_OPERAND_ROLES)} operands, got {roles!r}."
        )
    concepts = []
    for operand, allowed in zip(
        operands,
        ({"role", "concept"}, {"role", "concept", "period_factor"}),
        strict=True,
    ):
        unsupported = set(operand) - allowed
        if unsupported:
            raise ValueError(
                f"{prefix} operand {operand['role']!r} takes only "
                f"{sorted(allowed)!r}, got {sorted(unsupported)!r}."
            )
        concept = operand.get("concept")
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError(
                f"{prefix} operand {operand['role']!r} must name its concept."
            )
        concepts.append(concept)
    if concepts[0] == concepts[1]:
        raise ValueError(f"{prefix} count and mean must name different concepts.")
    factor = operands[1].get("period_factor", 1)
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(float(factor))
        or float(factor) <= 0
    ):
        raise ValueError(
            f"{prefix} period_factor must be a finite positive number, got {factor!r}."
        )


def _count_x_mean_operand_role(reference: LedgerTargetReference, fact: object) -> str:
    """The operand role whose concept the fact carries, or ``""``."""

    concepts = {_source_measure_concept(fact), _primary_measure_concept(fact)} - {""}
    roles = [
        str(operand["role"])
        for operand in reference.value_operands
        if str(operand["concept"]) in concepts
    ]
    return roles[0] if len(roles) == 1 else ""


def _count_x_mean_period_factor(reference: LedgerTargetReference) -> float:
    return float(reference.value_operands[1].get("period_factor", 1))


def _count_x_mean_cell_key(fact: object) -> tuple[str, ...]:
    """The published cell a count and its mean must share.

    Publishers file the two measures in sibling record sets
    (``...age_gender_type.month2025_05`` and ``...age_gender_type.mean.month2025_05``),
    so the record set is not part of the cell: its period, geography, entity,
    group value, dimensions and universe are.
    """

    return (
        _source_name(fact),
        _str_at(fact, "period", "type"),
        _str_at(fact, "period", "value"),
        _str_at(fact, "geography", "level"),
        _str_at(fact, "geography", "id"),
        _str_at(fact, "entity", "name"),
        _str_at(fact, "layout", "groupby_value_id"),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
        _domain(fact),
    )


def _resolve_monthly_window_count_x_mean_reference_facts(
    reference: LedgerTargetReference,
    matches: list[object],
) -> tuple[object, ...]:
    """One count and one mean fact per declared month, on one published cell.

    Returns them month by month, count before mean. Like the single-series
    windows, the guard also runs after direct-key resolution.
    """

    months = _declared_source_months(reference)
    prefix = (
        f"Ledger target reference {reference.name!r}: {MONTHLY_WINDOW_COUNT_X_MEAN}"
    )
    expected_count = len(COUNT_X_MEAN_OPERAND_ROLES) * len(months)
    if reference.expected_member_count not in (None, expected_count):
        raise ValueError(
            f"{prefix} expected_member_count disagrees with its "
            f"{len(months)} declared months."
        )
    grid: dict[tuple[str, str], object] = {}
    series: dict[str, set[tuple[str, ...]]] = {}
    for fact in matches:
        if not _fact_matches_selector(fact, reference.ledger_selector):
            raise ValueError(f"{prefix} resolved a fact outside its selector.")
        role = _count_x_mean_operand_role(reference, fact)
        if not role:
            raise ValueError(
                f"{prefix} selected a fact whose concept is neither operand's: "
                f"{_primary_measure_concept(fact)!r}."
            )
        month = _str_at(fact, "period", "value")
        if _str_at(fact, "period", "type") != "month" or month not in months:
            raise ValueError(
                f"{prefix} selected a fact outside the declared months: {month!r}."
            )
        if (month, role) in grid:
            raise ValueError(
                f"{prefix} matched more than one {role} fact for {month}; the "
                "selector must pin one published cell."
            )
        grid[(month, role)] = fact
        series.setdefault(role, set()).add(_selector_period_invariant_key(fact))
    missing = [
        f"{month} {role}"
        for month in months
        for role in COUNT_X_MEAN_OPERAND_ROLES
        if (month, role) not in grid
    ]
    if missing:
        raise ValueError(
            f"{prefix} requires one count and one mean fact for every declared "
            f"month; missing {missing!r}."
        )
    if any(len(keys) != 1 for keys in series.values()):
        raise ValueError(f"{prefix} matched more than one series for a role.")
    for month in months:
        if _count_x_mean_cell_key(grid[(month, "count")]) != _count_x_mean_cell_key(
            grid[(month, "mean")]
        ):
            raise ValueError(
                f"{prefix} pairs a count and a mean from different published "
                f"cells in {month}."
            )
    publications = {
        (_str_at(fact, "source_release_key"), _str_at(fact, "source", "source_sha256"))
        for fact in grid.values()
    }
    if len(publications) != 1 or not all(next(iter(publications))):
        raise ValueError(
            f"{prefix} requires every member to come from one publication with a "
            "nonempty source_release_key and source.source_sha256."
        )
    return tuple(
        grid[(month, role)] for month in months for role in COUNT_X_MEAN_OPERAND_ROLES
    )


def _monthly_window_count_x_mean_value(
    reference: LedgerTargetReference,
    facts: tuple[object, ...],
    numeric_values: list[float],
) -> tuple[float, dict[str, str]]:
    """Average over months of count x mean x period_factor, with every member."""

    factor = _count_x_mean_period_factor(reference)
    members = []
    products = []
    for index in range(0, len(facts), len(COUNT_X_MEAN_OPERAND_ROLES)):
        count_fact, mean_fact = facts[index], facts[index + 1]
        count, mean = numeric_values[index], numeric_values[index + 1]
        if count < 0 or mean < 0:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: "
                f"{MONTHLY_WINDOW_COUNT_X_MEAN} needs a non-negative count and "
                f"mean; got {count!r} and {mean!r} in "
                f"{_str_at(count_fact, 'period', 'value')}."
            )
        products.append(count * mean * factor)
        members.append(
            {
                "month": _str_at(count_fact, "period", "value"),
                "count": f"{count:.15g}",
                "mean": f"{mean:.15g}",
                "count_fact_key": _fact_key(count_fact)
                or _source_record_id(count_fact),
                "mean_fact_key": _fact_key(mean_fact) or _source_record_id(mean_fact),
            }
        )
    mean_unit = _str_at(facts[1], "observed_measure", "unit")
    return sum(products) / len(products), {
        "ledger_window_count_x_mean_count_concept": str(
            reference.value_operands[0]["concept"]
        ),
        "ledger_window_count_x_mean_mean_concept": str(
            reference.value_operands[1]["concept"]
        ),
        "ledger_window_count_x_mean_period_factor": f"{factor:.15g}",
        "ledger_window_count_x_mean_members": json.dumps(
            members, sort_keys=True, separators=(",", ":")
        ),
        # The value is count x mean x factor, not the mean fact's own unit.
        "ledger_measure_unit": f"{mean_unit} x {factor:.15g}"
        if mean_unit
        else f"x {factor:.15g}",
    }


_ALLOCATION_FIXED_PINS = (
    "source_name",
    "source_measure_id",
    "source_concept",
    "entity_name",
    "record_set_id",
    "period_type",
    "period_value",
    "geography_level",
    "geography_id",
    "geography_vintage",
)


def _validate_allocation_selector(
    selector: Mapping[str, object], *, prefix: str, grouped: bool = False
) -> None:
    missing = [
        key
        for key in _ALLOCATION_FIXED_PINS
        if selector.get(key) in (None, "")
        or isinstance(selector.get(key), (Mapping, list, tuple))
    ]
    dimensions = selector.get("dimensions")
    pins = selector.get("dimension_values", {})
    if (
        missing
        or not isinstance(dimensions, list)
        or len(set(dimensions)) != len(dimensions)
        or not isinstance(pins, Mapping)
        or set(dimensions) != set(pins)
    ):
        raise ValueError(
            f"{prefix} requires complete scalar source/geography/period and dimension pins; missing {missing!r}."
        )
    for value in pins.values():
        members = value if grouped and isinstance(value, list) else [value]
        if (
            not members
            or any(isinstance(member, (Mapping, list, tuple)) for member in members)
            or any(
                _strict_scalar_equal(member, previous)
                for index, member in enumerate(members)
                for previous in members[:index]
            )
        ):
            raise ValueError(
                f"{prefix} requires unique typed scalar dimension members."
            )


def _allocation_operand_selector(operand: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in operand.items() if key != "role"}


def validate_signed_discrepancy_policy(policy: object) -> dict[str, object]:
    """Validate the explicit, closed declaration permitting signed discrepancies."""
    keys = {"kind", "max_abs", "max_relative", "combine", "ruling", "disclosure"}
    if not isinstance(policy, Mapping) or set(policy) != keys:
        raise ValueError(
            "Signed discrepancy policy requires exactly its declared keys."
        )
    if policy["kind"] != "signed_proportional_reconciliation":
        raise ValueError("Unknown signed discrepancy policy kind.")
    if type(policy["max_abs"]) is not int or policy["max_abs"] < 0:
        raise ValueError("Signed discrepancy max_abs must be a nonnegative integer.")
    decimal = policy["max_relative"]
    if (
        not isinstance(decimal, str)
        or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", decimal) is None
    ):
        raise ValueError(
            "Signed discrepancy max_relative must be an exact decimal string."
        )
    if Fraction(decimal) > 1:
        raise ValueError("Signed discrepancy max_relative must be at most 1.")
    if policy["combine"] != "either":
        raise ValueError("Signed discrepancy bounds must combine as 'either'.")
    if any(
        not isinstance(policy[key], str) or not policy[key].strip()
        for key in ("ruling", "disclosure")
    ):
        raise ValueError("Signed discrepancy policy needs its ruling and disclosure.")
    return dict(policy)


def reconcile_signed_partition(
    partition: Mapping[str, Fraction],
    total: Fraction,
    policy: Mapping[str, object],
    *,
    region: str,
    positive_recipient_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, Fraction], dict[str, Fraction], tuple[str, ...]]:
    """Reconcile exact counts, optionally assigning a positive gap to one atom.

    The central allocation preserves every share, including for negative gaps.
    A positive-only recipient changes only the positive-gap allocation; zero
    and negative gaps continue to use central proportional reconciliation.
    """
    checked = validate_signed_discrepancy_policy(policy)
    if (
        not partition
        or any(not isinstance(key, str) or not key for key in partition)
        or not isinstance(total, Fraction)
        or any(not isinstance(value, Fraction) for value in partition.values())
        or total < 0
        or any(value < 0 for value in partition.values())
    ):
        raise ValueError(
            "Signed reconciliation requires nonnegative exact Fraction counts."
        )
    if positive_recipient_keys is not None and (
        len(positive_recipient_keys) != 1 or positive_recipient_keys[0] not in partition
    ):
        raise ValueError(
            "Positive discrepancy needs exactly one declared partition recipient."
        )
    leaves = sum(partition.values(), Fraction())
    residual = total - leaves
    if not leaves and total:
        raise ValueError(
            f"Signed reconciliation region {region}: R={residual}, L={leaves}; "
            "L=0 with a nonzero published total is refused."
        )
    bounds = tuple(
        name
        for name, passed in (
            ("absolute", abs(residual) <= checked["max_abs"]),
            (
                "relative",
                bool(leaves)
                and abs(residual) / leaves <= Fraction(checked["max_relative"]),
            ),
        )
        if passed
    )
    if not bounds:
        raise ValueError(
            f"Signed reconciliation region {region}: R={residual}, L={leaves} "
            f"exceeds either bound: max_abs={checked['max_abs']}, "
            f"max_relative={checked['max_relative']}."
        )
    if residual > 0 and positive_recipient_keys is not None:
        increments = {
            key: residual if key in positive_recipient_keys else Fraction()
            for key in sorted(partition)
        }
    else:
        increments = {
            key: value * residual / leaves if leaves else Fraction()
            for key, value in sorted(partition.items())
        }
    reconciled = {key: partition[key] + increments[key] for key in increments}
    if (
        any(value < 0 for value in reconciled.values())
        or sum(reconciled.values(), Fraction()) != total
    ):
        raise ValueError(f"Signed reconciliation region {region} failed exact closure.")
    return reconciled, increments, bounds


def _validate_allocation_reference(reference: LedgerTargetReference) -> None:
    prefix = f"LedgerTargetReference {reference.name!r}: {ALLOCATED_RESIDUAL_BY_RATIO}"
    operands = reference.value_operands
    roles = [
        operand.get("role") if isinstance(operand, Mapping) else None
        for operand in operands
    ]
    if roles not in (
        ["base", "total", "partition"],
        ["base", "total", "partition", "numerator", "denominator"],
    ):
        raise ValueError(
            f"{prefix} requires ordered base/total/partition and optional paired numerator/denominator operands."
        )
    base = operands[0]
    count = base.get("expected_member_count")
    if (
        set(base) != {"role", "expected_member_count"}
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or reference.expected_member_count not in (None, count)
    ):
        raise ValueError(
            f"{prefix} base requires a positive expected_member_count and inherits the reference selector without overrides."
        )
    _validate_allocation_selector(
        reference.ledger_selector, prefix=prefix, grouped=True
    )
    members = operands[2].get("member_selectors")
    recipients = operands[2].get("allocation_member_keys")
    partition_keys = {"role", "member_selectors", "allocation_member_keys"}
    signed = "discrepancy_policy" in operands[2]
    if signed:
        partition_keys.add("discrepancy_policy")
        validate_signed_discrepancy_policy(operands[2]["discrepancy_policy"])
        if "positive_allocation_member_keys" in operands[2]:
            partition_keys.add("positive_allocation_member_keys")
    if (
        set(operands[2]) != partition_keys
        or not isinstance(members, (list, tuple))
        or not members
    ):
        raise ValueError(
            f"{prefix} partition requires explicit member_selectors and allocation_member_keys."
        )
    selectors = [_allocation_operand_selector(operands[1]), *members]
    for selector in selectors:
        if not isinstance(selector, Mapping):
            raise ValueError(f"{prefix} member selectors must be mappings.")
        _validate_allocation_selector(selector, prefix=prefix)
        if (
            not isinstance(selector.get("aggregate_fact_key"), str)
            or not selector["aggregate_fact_key"]
        ):
            raise ValueError(
                f"{prefix} total and partition selectors require aggregate_fact_key pins."
            )
    keys = [selector["aggregate_fact_key"] for selector in members]
    if len(set(keys)) != len(keys) or selectors[0]["aggregate_fact_key"] in keys:
        raise ValueError(
            f"{prefix} partition members must be unique and exclude the total."
        )
    if (
        not isinstance(recipients, (list, tuple))
        or not recipients
        or any(not isinstance(key, str) for key in recipients)
        or len(set(recipients)) != len(recipients)
        or not set(recipients) <= set(keys)
    ):
        raise ValueError(
            f"{prefix} allocation_member_keys must be a nonempty unique subset of the partition."
        )
    if signed:
        if set(recipients) != set(keys):
            raise ValueError(
                f"{prefix} signed central recipients must be the full partition."
            )
        # Present means one recipient: an explicit null is not an absent key.
        positive = operands[2].get("positive_allocation_member_keys")
        if "positive_allocation_member_keys" in operands[2] and (
            not isinstance(positive, (list, tuple))
            or len(positive) != 1
            or not isinstance(positive[0], str)
            or positive[0] not in keys
        ):
            raise ValueError(
                f"{prefix} positive discrepancy needs exactly one partition recipient."
            )
    # Scalar dimensions identify the atoms independently of their fact keys.
    atomic_cells = [
        json.dumps(selector.get("dimension_values", {}), sort_keys=True)
        for selector in members
    ]
    if len(set(atomic_cells)) != len(atomic_cells):
        raise ValueError(f"{prefix} partition has duplicate atomic cells.")
    total = selectors[0]
    common = set(_ALLOCATION_FIXED_PINS) - {"record_set_id"}
    for selector in members:
        if any(selector[key] != total[key] for key in common):
            raise ValueError(
                f"{prefix} total and partition must share one source count universe, geography, vintage and period."
            )
        total_dimensions = set(total["dimensions"])
        atom_dimensions = set(selector["dimensions"])
        if not total_dimensions < atom_dimensions:
            raise ValueError(
                f"{prefix} total must be outside the atomic partition dimensions."
            )
        if atom_dimensions != set(members[0]["dimensions"]) or any(
            not _strict_scalar_equal(
                selector["dimension_values"][dimension],
                total["dimension_values"][dimension],
            )
            for dimension in total_dimensions
        ):
            raise ValueError(
                f"{prefix} partition dimensions and shared total dimension values must agree."
            )
    for operand in operands[3:]:
        _validate_allocation_selector(
            _allocation_operand_selector(operand), prefix=prefix
        )
    _validate_allocation_base_group(reference, members)


def _validate_allocation_base_group(
    reference: LedgerTargetReference, members: list[Mapping[str, object]]
) -> None:
    """Every typed cell requested by the base must belong to the partition."""
    selector = reference.ledger_selector
    pins = selector.get("dimension_values", {})
    dimensions = tuple(sorted(pins))
    rosters = tuple(
        pins[name] if isinstance(pins[name], list) else [pins[name]]
        for name in dimensions
    )
    cell_count = math.prod(len(roster) for roster in rosters)
    prefix = f"LedgerTargetReference {reference.name!r}: {ALLOCATED_RESIDUAL_BY_RATIO}"
    if reference.value_operands[0]["expected_member_count"] != cell_count:
        raise ValueError(
            f"{prefix} base expected_member_count must equal its explicit typed roster length."
        )
    if cell_count > len(members):
        raise ValueError(
            f"{prefix} base typed roster must be an exact subset of the atomic partition."
        )
    for cell in product(*rosters):
        matches = [
            member
            for member in members
            if all(member[key] == selector[key] for key in _ALLOCATION_FIXED_PINS)
            and set(member["dimensions"]) == set(dimensions)
            and all(
                _strict_scalar_equal(member["dimension_values"][name], value)
                for name, value in zip(dimensions, cell, strict=True)
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{prefix} base typed roster must be an exact subset of the atomic partition."
            )


def _allocation_auxiliary_selectors(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    return (
        _allocation_operand_selector(reference.value_operands[1]),
        *reference.value_operands[2]["member_selectors"],
        *(
            _allocation_operand_selector(operand)
            for operand in reference.value_operands[3:]
        ),
    )


def _allocation_universe_key(
    fact: object, partition_dimensions: set[str]
) -> tuple[object, ...]:
    # Category filters define the partition atoms, while all other filters
    # define the count universe and must agree with the published total.
    dimensions = _dimensions(fact)
    constraints = [
        row
        for row in _constraint_rows(fact)
        if not (
            isinstance(row, Mapping)
            and row.get("variable") in partition_dimensions
            and row.get("variable") in dimensions
            and row.get("operator") == "=="
            and row.get("role", "filter") == "filter"
            and _strict_scalar_equal(row.get("value"), dimensions[row["variable"]])
        )
    ]
    return (
        _domain(fact),
        json.dumps(constraints, sort_keys=True, separators=(",", ":")),
        _str_at(fact, "entity", "role"),
        _str_at(fact, "source_release_key"),
        _str_at(fact, "source", "source_sha256"),
        _str_at(fact, "source", "vintage"),
        json.dumps(_at(fact, "period_coverage"), sort_keys=True, separators=(",", ":")),
    )


def _resolve_allocation_reference_facts(
    reference: LedgerTargetReference,
    facts: tuple[object, ...],
    *,
    fact_index: _LedgerFactIndex | None = None,
    base_matches: list[object] | None = None,
) -> tuple[object, ...]:
    prefix = (
        f"Ledger target reference {reference.name!r}: {ALLOCATED_RESIDUAL_BY_RATIO}"
    )
    _validate_allocation_reference(reference)
    selected = []
    for selector in _allocation_auxiliary_selectors(reference):
        aggregate_key = selector.get("aggregate_fact_key")
        candidates = (
            fact_index.by_aggregate_fact_key.get(aggregate_key, ())
            if fact_index is not None
            and isinstance(aggregate_key, str)
            and aggregate_key
            else facts
        )
        matches = [
            fact for fact in candidates if _fact_matches_selector(fact, selector)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{prefix} each declared selector must resolve exactly once, got {len(matches)}."
            )
        selected.append(matches[0])
    total = selected[0]
    atoms = selected[1 : 1 + len(reference.value_operands[2]["member_selectors"])]
    base = (
        base_matches
        if base_matches is not None
        else [
            fact
            for fact in facts
            if _fact_matches_selector(fact, reference.ledger_selector)
        ]
    )
    keys = {_fact_key(fact) for fact in atoms}
    if len(base) != reference.value_operands[0]["expected_member_count"] or any(
        _fact_key(fact) not in keys for fact in base
    ):
        raise ValueError(
            f"{prefix} base must be its exact declared subset of the partition."
        )
    dimensions = set().union(*(set(_dimensions(fact)) for fact in atoms))
    universe = _allocation_universe_key(total, dimensions)
    if not _str_at(total, "source_release_key") or not _str_at(
        total, "source", "source_sha256"
    ):
        raise ValueError(
            f"{prefix} requires a pinned source snapshot (source_release_key and source_sha256)."
        )
    if any(_allocation_universe_key(fact, dimensions) != universe for fact in atoms):
        raise ValueError(
            f"{prefix} total and partition disagree on their count universe or source snapshot."
        )
    growth = selected[1 + len(atoms) :]
    if growth:
        base_geography = tuple(
            _str_at(total, "geography", key) for key in ("level", "id")
        )
        growth_geography = tuple(
            _str_at(growth[0], "geography", key) for key in ("level", "id")
        )
        if base_geography != growth_geography and not reference.metadata.get(
            "growth_geography_proxy"
        ):
            raise ValueError(
                f"{prefix} cross-geography growth requires a declared growth_geography_proxy."
            )
        _validated_growth_ratio(
            reference,
            total,
            *growth,
            exact="discrepancy_policy" in reference.value_operands[2],
        )
    for index, fact in enumerate(selected):
        if index <= len(atoms):
            _validate_resolved_reference_fact(fact, reference)
        else:
            _validate_window_ratio_operand_fact(fact, reference)
        read_value = (
            _exact_count_fact_value
            if "discrepancy_policy" in reference.value_operands[2]
            else _numeric_fact_value
        )
        if read_value(fact, reference) < 0:
            raise ValueError(f"{prefix} requires finite nonnegative counts.")
        _validate_fact_aggregation(fact, reference)
    labelled_facts = (
        *sorted(base, key=_fact_key),
        *sorted(atoms, key=_fact_key),
        total,
        *growth,
    )
    # A published cell may serve more than one operand role. Preserve each
    # fact once after checking every input selector's exact cardinality, but
    # never let a different fact that reuses a key replace a validated one:
    # every base, atom, total and growth fact enters this comparison.
    by_key: dict[str, object] = {}
    for fact in labelled_facts:
        key = _fact_key(fact)
        kept = by_key.setdefault(key, fact)
        if kept is not fact and kept != fact:
            raise ValueError(
                f"{prefix} distinct facts share aggregate_fact_key {key!r}."
            )
    return tuple(by_key.values())


def _signed_allocated_residual_value(
    reference: LedgerTargetReference, facts: tuple[object, ...]
) -> tuple[float, dict[str, str]]:
    """Compute signed allocation and its exact receipt before the float API boundary."""
    partition = reference.value_operands[2]
    facts_by_key = {_fact_key(fact): fact for fact in facts}
    atom_keys = sorted(
        selector["aggregate_fact_key"] for selector in partition["member_selectors"]
    )
    atoms = {key: facts_by_key[key] for key in atom_keys}
    total = facts_by_key[reference.value_operands[1]["aggregate_fact_key"]]
    region = _str_at(total, "geography", "id")
    counts = {
        key: _exact_count_fact_value(fact, reference) for key, fact in atoms.items()
    }
    total_value = _exact_count_fact_value(total, reference)
    positive_keys = partition.get("positive_allocation_member_keys")
    adjusted, increments, bounds = reconcile_signed_partition(
        counts,
        total_value,
        partition["discrepancy_policy"],
        region=region,
        positive_recipient_keys=None if positive_keys is None else tuple(positive_keys),
    )
    stated = sum(counts.values(), Fraction())
    residual = total_value - stated
    recipients = sorted(
        positive_keys
        if residual > 0 and positive_keys is not None
        else partition["allocation_member_keys"]
    )
    base_keys = sorted(
        key
        for key, fact in atoms.items()
        if _fact_matches_selector(fact, reference.ledger_selector)
    )
    growth = tuple(
        next(
            fact
            for fact in facts
            if _fact_matches_selector(fact, _allocation_operand_selector(operand))
        )
        for operand in reference.value_operands[3:]
    )
    ratio = (
        _validated_growth_ratio(reference, total, *growth, exact=True)
        if growth
        else Fraction(1)
    )
    final_targets = {key: value * ratio for key, value in adjusted.items()}
    value = sum((final_targets[key] for key in base_keys), Fraction())

    def rational(number: Fraction) -> dict[str, int]:
        return {"numerator": number.numerator, "denominator": number.denominator}

    labelled_keys = {
        "base": base_keys,
        "partition": atom_keys,
        "recipients": recipients,
        "total": _fact_key(total),
        **(
            {"numerator": _fact_key(growth[0]), "denominator": _fact_key(growth[1])}
            if growth
            else {}
        ),
    }
    policy = dict(partition["discrepancy_policy"])
    receipt = {
        "operation": ALLOCATED_RESIDUAL_BY_RATIO,
        "version": 2,
        "ruling": policy["ruling"],
        "policy": policy,
        "region": region,
        "selected_profile": reference.metadata.get(
            "family_discrepancy",
            reference.metadata.get("residual_allocation_profile", ""),
        ),
        "bounds_passed": list(bounds),
        "fact_keys": labelled_keys,
        "source_snapshot": {
            "source_release_key": _str_at(total, "source_release_key"),
            "source_sha256": _str_at(total, "source", "source_sha256"),
            "source_vintage": _str_at(total, "source", "vintage"),
        },
        "growth_geography_proxy": reference.metadata.get("growth_geography_proxy", ""),
        "x": {key: rational(number) for key, number in counts.items()},
        "F": rational(total_value),
        "L": rational(stated),
        "R": rational(residual),
        "increments": {key: rational(number) for key, number in increments.items()},
        "reconciled": {key: rational(number) for key, number in adjusted.items()},
        "growth_operands": {
            role: {
                "fact_key": _fact_key(fact),
                "value": rational(_exact_count_fact_value(fact, reference)),
            }
            for role, fact in zip(
                ("numerator", "denominator"), growth, strict=bool(growth)
            )
        },
        "Q": rational(ratio),
        "final_targets": {
            key: rational(number) for key, number in final_targets.items()
        },
        "source_period": _comparable_period_value(total),
        "target_period": reference.period,
        "final_target": rational(value),
    }
    return float(value), {
        "ledger_allocation_receipt": json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "ledger_value_base": str(sum((counts[key] for key in base_keys), Fraction())),
        "ledger_value_ratio": str(ratio),
    }


def _allocated_residual_value(
    reference: LedgerTargetReference, facts: tuple[object, ...]
) -> tuple[float, dict[str, str]]:
    partition = reference.value_operands[2]
    if "discrepancy_policy" in partition:
        return _signed_allocated_residual_value(reference, facts)
    facts_by_key = {_fact_key(fact): fact for fact in facts}
    atoms = tuple(
        facts_by_key[key]
        for key in sorted(
            selector["aggregate_fact_key"] for selector in partition["member_selectors"]
        )
    )
    total = facts_by_key[reference.value_operands[1]["aggregate_fact_key"]]
    base = [
        fact
        for fact in atoms
        if _fact_matches_selector(fact, reference.ledger_selector)
    ]
    recipients = set(partition["allocation_member_keys"])
    total_value = _numeric_fact_value(total, reference)
    stated = math.fsum(_numeric_fact_value(fact, reference) for fact in atoms)
    base_value = math.fsum(_numeric_fact_value(fact, reference) for fact in base)
    recipient_mass = math.fsum(
        _numeric_fact_value(fact, reference)
        for fact in atoms
        if _fact_key(fact) in recipients
    )
    recipient_base = math.fsum(
        _numeric_fact_value(fact, reference)
        for fact in base
        if _fact_key(fact) in recipients
    )
    residual = total_value - stated
    if not math.isfinite(residual) or residual < 0:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: negative or nonfinite residual {residual!r} is refused."
        )
    if residual and not recipient_mass > 0:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: positive residual requires positive recipient mass."
        )
    increment = residual * (recipient_base / recipient_mass) if residual else 0.0
    growth = tuple(
        next(
            fact
            for fact in facts
            if _fact_matches_selector(fact, _allocation_operand_selector(operand))
        )
        for operand in reference.value_operands[3:]
    )
    ratio = _validated_growth_ratio(reference, total, *growth) if growth else 1.0
    value = (base_value + increment) * ratio
    if not math.isfinite(value):
        raise ValueError(
            f"Ledger target reference {reference.name!r}: allocation produced nonfinite value."
        )
    labelled_keys = {
        "base": sorted(_fact_key(fact) for fact in base),
        "partition": sorted(_fact_key(fact) for fact in atoms),
        "recipients": sorted(recipients),
        "total": _fact_key(total),
        **(
            {"numerator": _fact_key(growth[0]), "denominator": _fact_key(growth[1])}
            if growth
            else {}
        ),
    }
    receipt = {
        "operation": ALLOCATED_RESIDUAL_BY_RATIO,
        "version": 1,
        "ruling": reference.metadata.get("allocation_ruling", ""),
        "selected_profile": reference.metadata.get(
            "census_residual_allocation",
            reference.metadata.get("residual_allocation_profile", ""),
        ),
        "fact_keys": labelled_keys,
        "source_snapshot": {
            "source_release_key": _str_at(total, "source_release_key"),
            "source_sha256": _str_at(total, "source", "source_sha256"),
            "source_vintage": _str_at(total, "source", "vintage"),
        },
        "growth_geography_proxy": reference.metadata.get("growth_geography_proxy", ""),
        "member_digests": {
            role: hashlib.sha256(
                json.dumps(labelled_keys[role], separators=(",", ":")).encode()
            ).hexdigest()
            for role in ("base", "partition", "recipients")
        },
        "T": total_value,
        "L": stated,
        "R": residual,
        "B": base_value,
        "A": recipient_mass,
        "W": recipient_base,
        "allocated_increment": increment,
        "N": _numeric_fact_value(growth[0], reference) if growth else None,
        "D": _numeric_fact_value(growth[1], reference) if growth else None,
        "Q": ratio,
        "zero_residual": residual == 0,
        "source_period": _comparable_period_value(total),
        "target_period": reference.period,
        "final_target": value,
    }
    return value, {
        "ledger_allocation_receipt": json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "ledger_value_base": repr(base_value),
        "ledger_value_ratio": repr(ratio),
    }


def _scaled_by_ratio_operand_selectors(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    """The numerator and denominator selectors, with the base pins inherited."""

    inherited = {
        key: reference.ledger_selector[key]
        for key in _SCALED_BY_RATIO_INHERITED_SELECTOR_KEYS
        if key in reference.ledger_selector
    }
    return tuple(
        {
            **inherited,
            **{str(key): value for key, value in operand.items() if key != "role"},
        }
        for operand in reference.value_operands[1:]
    )


def _rolled_forward_operand_selectors(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    """The numerator and denominator selectors, each with its own period pin."""

    inherited = {
        key: reference.ledger_selector[key]
        for key in ("source_name", "assertion")
        if key in reference.ledger_selector
    }
    return tuple(
        {
            **inherited,
            **{str(key): value for key, value in operand.items() if key != "role"},
        }
        for operand in reference.value_operands[1:]
    )


def _rolled_forward_base_overlay(reference: LedgerTargetReference) -> dict[str, object]:
    return {
        str(key): value
        for key, value in reference.value_operands[0].items()
        if key not in {"role", "expected_member_count"}
    }


def _rolled_forward_base_selector(
    reference: LedgerTargetReference,
) -> Mapping[str, object]:
    return {
        **dict(reference.ledger_selector),
        **_rolled_forward_base_overlay(reference),
    }


def _resolve_rolled_forward_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
    facts: tuple[object, ...],
) -> tuple[object, ...]:
    """The base cells at their latest period, then the two pinned operands.

    The base is every eligible cell at the latest period the selector reaches
    (one cell, or a set the operation sums); a base operand that moves the
    geography (a nation's estimate built from its districts' census cells)
    selects below the reference's own geography, which the spec keeps. The
    numerator must resolve at the target period and the denominator at the
    base's own period, each exactly once, so the ratio is the growth between
    the two.
    """

    if _rolled_forward_base_overlay(reference):
        base_selector = _rolled_forward_base_selector(reference)
        eligible_matches = _eligible_selector_matches(
            reference,
            [fact for fact in facts if _fact_matches_selector(fact, base_selector)],
        )
    if not eligible_matches:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: "
            f"{ROLLED_FORWARD_BY_RATIO} base matched no eligible fact."
        )
    latest = max(_period_key(fact) for fact in eligible_matches)
    base = [fact for fact in eligible_matches if _period_key(fact) == latest]
    _assert_rolled_forward_base_cells(reference, base)
    resolved: list[object] = list(base)
    for operand, selector in zip(
        reference.value_operands[1:],
        _rolled_forward_operand_selectors(reference),
        strict=True,
    ):
        matches = [
            fact
            for fact in facts
            if _fact_matches_selector(fact, selector)
            and _assertion_allowed(reference, fact)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: "
                f"{ROLLED_FORWARD_BY_RATIO} operand {operand.get('role')!r} "
                f"must resolve exactly once, got {len(matches)} matches."
            )
        resolved.append(matches[0])
    _validated_growth_ratio(reference, base[0], resolved[-2], resolved[-1])
    return tuple(resolved)


def _assert_rolled_forward_base_cells(
    reference: LedgerTargetReference, base: list[object]
) -> None:
    """Require an exact scalar, geography roster, or typed dimension roster.

    Summing whatever matches at the latest period would count a duplicate
    fact twice and pass a missing district silently.
    """

    # An explicit null on the base operand falls back to the reference's own
    # count, as the roster validator does; it never switches the check off.
    count = reference.value_operands[0].get("expected_member_count")
    if count is None:
        count = reference.expected_member_count
    if count is not None and len(base) != count:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: "
            f"{ROLLED_FORWARD_BY_RATIO} expected {count} base facts, got {len(base)}."
        )
    roster = _rolled_forward_grouped_member_roster(reference)
    if roster:
        selector = _rolled_forward_base_selector(reference)
        dimension, members = roster
        found = [_dimensions(fact).get(dimension) for fact in base]
        if any(not _fact_matches_selector(fact, selector) for fact in base) or any(
            sum(_strict_scalar_equal(value, member) for value in found) != 1
            for member in members
        ):
            raise ValueError(
                f"Ledger target reference {reference.name!r}: "
                f"{ROLLED_FORWARD_BY_RATIO} base must resolve exactly one fact "
                f"for each declared typed member of {dimension!r}: {members!r}; "
                f"got {found!r}."
            )
        return
    declared = None
    if _rolled_forward_base_overlay(reference):
        declared = _rolled_forward_base_selector(reference).get("geography_id")
    if isinstance(declared, (list, tuple)):
        expected = sorted(str(value) for value in declared)
        found = sorted(_str_at(fact, "geography", "id") for fact in base)
        if found != expected:
            missing = sorted(set(expected) - set(found))
            repeated = sorted({area for area in found if found.count(area) > 1})
            raise ValueError(
                f"Ledger target reference {reference.name!r}: "
                f"{ROLLED_FORWARD_BY_RATIO} base must resolve exactly one cell "
                f"for each of its {len(expected)} geographies at its latest "
                f"period; missing {missing}, repeated {repeated}."
            )
        return
    if len(base) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: "
            f"{ROLLED_FORWARD_BY_RATIO} base must resolve exactly one cell at "
            f"its latest period, got {len(base)}."
        )


def _rolled_forward_grouped_member_roster(
    reference: LedgerTargetReference,
) -> tuple[str, list[object]] | None:
    """Validate the narrow, one-geography dimension-member sum contract."""

    prefix = f"LedgerTargetReference {reference.name!r}: {ROLLED_FORWARD_BY_RATIO}"
    operand_count = reference.value_operands[0].get("expected_member_count")
    if operand_count is not None and reference.expected_member_count not in (
        None,
        operand_count,
    ):
        raise ValueError(f"{prefix}: base expected_member_count declarations disagree.")
    selector = _rolled_forward_base_selector(reference)
    pins = selector.get("dimension_values")
    if not isinstance(pins, Mapping):
        return None
    rosters = [
        (str(key), value) for key, value in pins.items() if isinstance(value, list)
    ]
    if not rosters:
        return None
    if len(rosters) != 1:
        raise ValueError(f"{prefix}: grouped base requires one typed member roster.")
    dimension, members = rosters[0]
    count = (
        operand_count if operand_count is not None else reference.expected_member_count
    )
    if not members or count != len(members):
        raise ValueError(
            f"{prefix}: grouped base expected_member_count must equal its explicit "
            "typed member roster length (base facts only)."
        )
    if any(isinstance(member, (Mapping, list, tuple)) for member in members) or any(
        _strict_scalar_equal(member, previous)
        for index, member in enumerate(members)
        for previous in members[:index]
    ):
        raise ValueError(
            f"{prefix}: grouped base typed member roster must be unique scalars."
        )
    fixed = (
        "source_name",
        "source_measure_id",
        "source_concept",
        "entity_name",
        "record_set_id",
        "period_type",
        "period_value",
        "geography_level",
        "geography_id",
        "geography_vintage",
    )
    missing = [
        key
        for key in fixed
        if selector.get(key) in (None, "")
        or isinstance(selector.get(key), (Mapping, list, tuple))
    ]
    dimensions = selector.get("dimensions")
    if (
        missing
        or not isinstance(dimensions, list)
        or len(dimensions) != len(set(dimensions))
        or set(dimensions) != set(pins)
    ):
        raise ValueError(
            f"{prefix}: grouped base must fix one source, entity, geography, "
            "record set and source period, and its complete dimension-name set; "
            f"missing or non-scalar pins {missing!r}."
        )
    return dimension, members


def _rolled_forward_series_key(fact: object) -> tuple[str, ...]:
    """What makes two facts one series at two periods.

    Narrower than the selector's period-invariant key on purpose: a publisher
    issues a period's estimate and the revision of the period before it as
    separate tables of one release (ONS mid-2025 beside revised mid-2024), so
    the record set and measure id may differ while the series is the same.
    """

    return (
        _source_name(fact),
        _source_measure_concept(fact),
        _str_at(fact, "geography", "level"),
        _str_at(fact, "geography", "id"),
        _str_at(fact, "entity", "name"),
        _str_at(fact, "aggregation", "method"),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
        _domain(fact),
    )


def reference_fact_selectors(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    """Every selector a reference resolves facts through.

    The reference's own selector first; a ``scaled_by_ratio`` reference adds
    its numerator and denominator selectors, which lie outside the base cell
    (national facts for a region-pinned cell). Callers that pre-filter the
    facts they hand to :func:`compile_ledger_target_references` must keep a
    fact matching any of these, or the quotient can never resolve.
    """

    selectors: list[Mapping[str, object]] = []
    if reference.ledger_selector:
        selectors.append(dict(reference.ledger_selector))
    if reference.value_operation == "scaled_by_ratio":
        selectors.extend(_scaled_by_ratio_operand_selectors(reference))
    if reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO:
        selectors.extend(_window_ratio_operand_selectors(reference))
    if reference.value_operation == ROLLED_FORWARD_BY_RATIO:
        if _rolled_forward_base_overlay(reference):
            selectors.append(_rolled_forward_base_selector(reference))
        selectors.extend(_rolled_forward_operand_selectors(reference))
    if reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
        selectors.extend(_allocation_auxiliary_selectors(reference))
    return tuple(selectors)


def _resolve_scaled_by_ratio_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
    facts: tuple[object, ...],
) -> tuple[object, ...]:
    """Resolve the base cell from the selector and the quotient from the feed.

    The numerator and denominator are national facts outside the cell's own
    selector (a region-pinned selector never matches a UK-level row), so they
    are matched over every fact the compile was given, with the base
    selector's source and period pins inherited unless the operand overrides
    them. All three must resolve exactly once at the same period.
    """

    base = _latest_period_selector_match(reference, eligible_matches)
    if base is None:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: scaled_by_ratio base "
            f"cell did not resolve exactly once ({len(eligible_matches)} "
            "eligible matches)."
        )
    resolved: list[object] = [base]
    for operand, selector in zip(
        reference.value_operands[1:],
        _scaled_by_ratio_operand_selectors(reference),
        strict=True,
    ):
        matches = [fact for fact in facts if _fact_matches_selector(fact, selector)]
        eligible = _eligible_selector_matches(reference, matches)
        match = _latest_period_selector_match(reference, eligible)
        if match is None:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: scaled_by_ratio "
                f"operand {operand.get('role')!r} did not resolve exactly once "
                f"({len(eligible)} eligible matches)."
            )
        resolved.append(match)
    periods = {_period_key(fact) for fact in resolved}
    if len(periods) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: scaled_by_ratio "
            "operands must resolve at the same period as the base cell."
        )
    return tuple(resolved)


def _validate_linear_combination_operands(
    name: str, operands: tuple[Mapping[str, object], ...]
) -> None:
    """A signed combination needs at least one weighted, selector-bearing operand."""

    if not operands:
        raise ValueError(
            f"LedgerTargetReference {name!r}: linear_combination requires at "
            "least one weighted operand."
        )
    for index, operand in enumerate(operands):
        if not isinstance(operand, Mapping):
            raise ValueError(
                f"LedgerTargetReference {name!r}: linear_combination operand "
                f"{index} must be a mapping."
            )
        weight = operand.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight == 0
        ):
            raise ValueError(
                f"LedgerTargetReference {name!r}: linear_combination operand "
                f"{index} needs a finite nonzero numeric weight."
            )
        count = operand.get("expected_member_count")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
        ):
            raise ValueError(
                f"LedgerTargetReference {name!r}: linear_combination operand "
                f"{index} expected_member_count must be a positive integer."
            )
        dimensions = operand.get("dimension_values")
        if dimensions is not None and not isinstance(dimensions, Mapping):
            raise ValueError(
                f"LedgerTargetReference {name!r}: linear_combination operand "
                f"{index} dimension_values must be a mapping."
            )
        if not any(key not in LINEAR_COMBINATION_OPERAND_KEYS for key in operand) and (
            dimensions is None
        ):
            raise ValueError(
                f"LedgerTargetReference {name!r}: linear_combination operand "
                f"{index} declares no selector overlay; every operand must "
                "narrow the shared selector."
            )


def _linear_combination_operand_selector(
    reference: LedgerTargetReference,
    operand: Mapping[str, object],
) -> dict[str, object]:
    """The shared selector overlaid by one operand's own keys."""

    shared = reference.ledger_selector
    selector: dict[str, object] = {
        **dict(shared),
        **{
            str(key): value
            for key, value in operand.items()
            if key not in LINEAR_COMBINATION_OPERAND_KEYS
        },
    }
    shared_dimensions = shared.get("dimension_values")
    dimensions = operand.get("dimension_values")
    if isinstance(dimensions, Mapping) or isinstance(shared_dimensions, Mapping):
        selector["dimension_values"] = {
            **(
                dict(shared_dimensions)
                if isinstance(shared_dimensions, Mapping)
                else {}
            ),
            **(dict(dimensions) if isinstance(dimensions, Mapping) else {}),
        }
    return selector


def _linear_combination_operand_label(operand: Mapping[str, object], index: int) -> str:
    label = operand.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    parts = [
        str(operand[key])
        for key in ("source_measure_id", "source_concept")
        if isinstance(operand.get(key), str) and operand.get(key)
    ]
    dimensions = operand.get("dimension_values")
    if isinstance(dimensions, Mapping) and dimensions:
        parts.append(
            "["
            + ",".join(f"{key}={value}" for key, value in sorted(dimensions.items()))
            + "]"
        )
    return "".join(parts) or f"operand_{index}"


def _linear_combination_formula(reference: LedgerTargetReference) -> str:
    terms = []
    for index, operand in enumerate(reference.value_operands):
        weight = float(operand["weight"])
        sign = "+" if weight > 0 else "-"
        magnitude = abs(weight)
        scale = "" if magnitude == 1 else f"{magnitude:g}*"
        terms.append(
            f"{sign}{scale}{_linear_combination_operand_label(operand, index)}"
        )
    return " ".join(terms)


def _resolve_linear_combination_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    """Resolve every weighted operand at one shared latest period.

    Each operand narrows the shared selector; its members are the facts of
    the latest period partition, and a multi-member operand must declare
    ``expected_member_count`` (the sum over those members is what the weight
    scales).  No fact may belong to two operands, and every operand must land
    on the same period, so the combination is a statement about one
    publication vintage.
    """

    resolved: list[object] = []
    claimed: dict[int, int] = {}
    operand_periods: list[tuple[int, int, str]] = []
    for index, operand in enumerate(reference.value_operands):
        label = _linear_combination_operand_label(operand, index)
        selector = _linear_combination_operand_selector(reference, operand)
        matches = [
            fact for fact in eligible_matches if _fact_matches_selector(fact, selector)
        ]
        if not matches:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: linear_combination "
                f"operand {label!r} matched no eligible Ledger fact."
            )
        partitions: dict[tuple[int, int, str], list[object]] = {}
        for fact in matches:
            partitions.setdefault(
                _reference_period_partition_key(fact, reference), []
            ).append(fact)
        latest_period = max(partitions)
        members = partitions[latest_period]
        expected = operand.get("expected_member_count")
        if expected is None and len(members) != 1:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: linear_combination "
                f"operand {label!r} resolved {len(members)} facts at the latest "
                "period; declare expected_member_count to sum a member set."
            )
        if expected is not None and len(members) != expected:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: linear_combination "
                f"operand {label!r} expected {expected} members at the latest "
                f"period but resolved {len(members)}; a declared member is missing."
            )
        for fact in members:
            if id(fact) in claimed:
                raise ValueError(
                    f"Ledger target reference {reference.name!r}: linear_combination "
                    f"operand {label!r} claims a fact operand "
                    f"{claimed[id(fact)]} already resolved; operand selectors "
                    "must be disjoint."
                )
            claimed[id(fact)] = index
        operand_periods.append(latest_period)
        resolved.extend(
            sorted(members, key=lambda fact: _fact_key(fact) or _source_record_id(fact))
        )
    if len(set(operand_periods)) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: linear_combination "
            "operands must resolve at the same latest period."
        )
    if (
        reference.expected_member_count is not None
        and len(resolved) != reference.expected_member_count
    ):
        raise ValueError(
            f"Ledger target reference {reference.name!r}: linear_combination "
            f"expected {reference.expected_member_count} member facts in total "
            f"but resolved {len(resolved)}."
        )
    return tuple(resolved)


def _linear_combination_weights(
    reference: LedgerTargetReference,
    facts: tuple[object, ...],
) -> tuple[float, ...]:
    """The weight of the one operand each resolved fact belongs to."""

    selectors = [
        (
            float(operand["weight"]),
            _linear_combination_operand_selector(reference, operand),
        )
        for operand in reference.value_operands
    ]
    weights: list[float] = []
    for fact in facts:
        owners = [
            weight
            for weight, selector in selectors
            if _fact_matches_selector(fact, selector)
        ]
        if len(owners) != 1:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: a resolved fact "
                f"belongs to {len(owners)} linear_combination operands; expected "
                "exactly one."
            )
        weights.append(owners[0])
    return tuple(weights)


def _declared_source_months(reference: LedgerTargetReference) -> tuple[str, ...]:
    months = reference.ledger_selector.get("period_value")
    if (
        reference.ledger_selector.get("period_type") != "month"
        or not isinstance(months, list)
        or not months
        or not all(
            isinstance(month, str)
            and re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month)
            for month in months
        )
        or months != sorted(set(months))
    ):
        raise ValueError(
            "A monthly window requires period_type=month and explicit unique ordered YYYY-MM period_value months."
        )
    return tuple(months)


def _monthly_window_operands(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    if reference.value_operation == "monthly_window_average":
        if reference.value_operands:
            raise ValueError(
                "A single-series monthly window does not accept value_operands."
            )
        return ()
    dimensions = []
    for operand in reference.value_operands:
        if not isinstance(operand, Mapping) or set(operand) != {"dimension_values"}:
            raise ValueError(
                "A monthly window sum operand requires only dimension_values."
            )
        values = operand["dimension_values"]
        if (
            not isinstance(values, Mapping)
            or not values
            or any(
                not isinstance(key, str)
                or not key
                or value is None
                or isinstance(value, (Mapping, list, tuple, set))
                for key, value in values.items()
            )
        ):
            raise ValueError(
                "A monthly window sum operand requires complete scalar dimension_values."
            )
        dimensions.append(values)
    if not dimensions or any(set(value) != set(dimensions[0]) for value in dimensions):
        raise ValueError(
            "A monthly window sum requires operands with the same complete dimension keys."
        )
    encodings = [
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in dimensions
    ]
    if len(set(encodings)) != len(encodings):
        raise ValueError("A monthly window sum requires disjoint operand tuples.")
    return tuple(dimensions)


def _monthly_window_source_key(
    fact: object, dimensions: Mapping[str, object]
) -> tuple[str, ...]:
    """All window members must share a publication and statistical measure."""
    return (
        _source_name(fact),
        _str_at(fact, "source_release_key"),
        _str_at(fact, "source", "source_sha256"),
        _str_at(fact, "source", "vintage"),
        _str_at(fact, "source", "source_file"),
        _str_at(fact, "source", "source_table")
        or _str_at(fact, "observed_measure", "source_table"),
        _str_at(fact, "observed_measure", "source_measure_id")
        or _str_at(fact, "layout", "measure_id"),
        _source_measure_concept(fact),
        _str_at(fact, "observed_measure", "unit"),
        json.dumps(_at(fact, "entity"), sort_keys=True),
        json.dumps(_at(fact, "geography"), sort_keys=True),
        # Neither the record set nor its selector spec id: a publisher can
        # keep an established spec id on one month (DWP's May 2025 UC cells,
        # PolicyEngine/chronicle#314) and publish each summed cell as its own
        # record set (the UC child counts) while the release, raw file and
        # table above are one. The per-cell series check still pins each
        # cell to one record set across its months.
        _str_at(fact, "layout", "groupby_dimension"),
        _str_at(fact, "aggregation", "method"),
        _domain(fact),
        json.dumps(
            [
                constraint
                for constraint in _constraint_rows(fact)
                if _str_at(constraint, "variable") not in dimensions
                and _str_at(constraint, "operator") not in _BAND_EDGE_OPERATORS
            ],
            sort_keys=True,
        ),
    )


#: A summed cell may be a published band whose edges the publisher also states
#: as numeric bounds (a single year of age 66 carries ``age >= 66`` and
#: ``age < 67``). Those edges belong to the cell, like its dimension value, so
#: they differ between cells; each cell's own months are still compared on
#: every constraint by the one-series check (microcosm#1069).
_BAND_EDGE_OPERATORS = frozenset(("<", "<=", ">", ">="))


def _resolve_monthly_window_reference_facts(
    reference: LedgerTargetReference,
    matches: list[object],
) -> tuple[object, ...]:
    """Require a complete month-by-cell grid before averaging monthly totals.

    The same guard runs after direct-key resolution, so identifiers cannot
    bypass the explicit window, selector, or operand coverage contract.
    """
    months = _declared_source_months(reference)
    operands = _monthly_window_operands(reference)
    cell_count = len(operands) or 1
    expected_count = len(months) * cell_count
    prefix = f"Ledger target reference {reference.name!r}: monthly window"
    if reference.expected_member_count not in (None, expected_count):
        raise ValueError(
            f"{prefix} expected_member_count disagrees with its declared month-by-cell grid."
        )
    if len(matches) != expected_count or any(
        not _fact_matches_selector(fact, reference.ledger_selector) for fact in matches
    ):
        raise ValueError(
            f"{prefix} requires exactly {expected_count} selected facts, one per declared month and cell; resolved {len(matches)}."
        )
    grid: dict[tuple[str, int], object] = {}
    partitions: dict[int, set[tuple[str, ...]]] = {}
    for fact in matches:
        publication = _at(fact, "source_release_key")
        raw_hash = _at(fact, "source", "source_sha256")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (publication, raw_hash)
        ):
            raise ValueError(
                f"{prefix} requires nonempty source_release_key and "
                "source.source_sha256 on every member."
            )
        if operands:
            cells = [
                index
                for index, values in enumerate(operands)
                if set(_dimensions(fact)) == set(values)
                and _dimension_values_match(fact, values)
            ]
        else:
            cells = [0]
        if len(cells) != 1:
            raise ValueError(
                f"{prefix} requires every fact to match exactly one complete operand tuple."
            )
        cell = cells[0]
        key = (_str_at(fact, "period", "value"), cell)
        if key in grid:
            raise ValueError(f"{prefix} contains a duplicate month/cell observation.")
        grid[key] = fact
        partitions.setdefault(cell, set()).add(_selector_period_invariant_key(fact))
    if any(len(series) != 1 for series in partitions.values()):
        raise ValueError(
            f"{prefix} matched multiple semantic series for a declared cell."
        )
    dimension_cells = operands[0] if operands else {}
    if (
        len({_monthly_window_source_key(fact, dimension_cells) for fact in matches})
        != 1
    ):
        raise ValueError(
            f"{prefix} cells must share source publication, measure, entity and geography."
        )
    expected_grid = [(month, cell) for month in months for cell in range(cell_count)]
    if set(grid) != set(expected_grid):
        raise ValueError(f"{prefix} is missing a declared month/cell observation.")
    member_keys = [_fact_key(fact) or _source_record_id(fact) for fact in matches]
    if not all(member_keys) or len(set(member_keys)) != len(member_keys):
        raise ValueError(
            f"{prefix} requires distinct non-empty member fact identities."
        )
    return tuple(grid[key] for key in expected_grid)


def _resolve_calendar_year_average_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    monthly_matches = _monthly_operation_matches(
        reference, eligible_matches, operation="calendar_year_average"
    )
    target_year = _calendar_year_from_reference(reference)
    year_matches = [
        fact for fact in monthly_matches if _fact_year_month(fact)[0] == target_year
    ]
    if not year_matches:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'calendar_year_average' found no monthly Ledger facts in calendar "
            f"year {target_year}."
        )
    return _latest_series_partition(
        reference, year_matches, operation="calendar_year_average"
    )


def _validate_calendar_year_window_reference(
    reference: LedgerTargetReference,
) -> None:
    if reference.value_operands:
        raise ValueError(
            f"LedgerTargetReference {reference.name!r}: calendar_year_window "
            "takes no value_operands; the two overlapping years and their "
            "month weights are the operation's own definition."
        )
    if reference.period_match_policy != "latest_not_after":
        raise ValueError(
            f"LedgerTargetReference {reference.name!r}: calendar_year_window "
            "resolves the two years overlapping the target calendar year itself; "
            "declare period_match_policy 'latest_not_after' (the default)."
        )
    period_key = _period_key_from_value(reference.period)
    if reference.period is None or not period_key[0] or period_key[1] % 100 != 99:
        raise ValueError(
            f"LedgerTargetReference {reference.name!r}: calendar_year_window "
            "requires an explicit calendar-year target period."
        )
    measure_map = reference.ledger_selector.get(SOURCE_MEASURE_ID_BY_OPENING_YEAR)
    if SOURCE_TABLE_BY_OPENING_YEAR in reference.ledger_selector:
        if measure_map is not None:
            raise ValueError(
                f"LedgerTargetReference {reference.name!r}: calendar_year_window "
                f"declares both {SOURCE_MEASURE_ID_BY_OPENING_YEAR!r} and "
                f"{SOURCE_TABLE_BY_OPENING_YEAR!r}; it maps its years by measure "
                "or by table, not both."
            )
        _validate_table_by_opening_year(reference, "calendar_year_window")
    if measure_map is None:
        return
    for key in ("source_measure_id", "layout_measure_id"):
        if reference.ledger_selector.get(key) not in (None, ""):
            raise ValueError(
                f"LedgerTargetReference {reference.name!r}: calendar_year_window "
                f"declares both {key!r} and {SOURCE_MEASURE_ID_BY_OPENING_YEAR!r}; "
                "the map names each year's measure itself."
            )
    if _calendar_year_window_series_count(reference) != 1:
        raise ValueError(
            f"LedgerTargetReference {reference.name!r}: calendar_year_window "
            f"reads {SOURCE_MEASURE_ID_BY_OPENING_YEAR!r} as one series; it "
            "cannot be combined with a list-valued series key or several "
            "pinned dimension values."
        )
    # The map's years are checked against the window when it resolves, not
    # here: a compile may restamp the reference to another period, and a map
    # without that period's years then defers like any reference without facts.
    _measure_id_by_opening_year(measure_map)


def _validate_table_by_opening_year(
    reference: LedgerTargetReference, operation: str
) -> None:
    """A window that maps its years to tables names no other table.

    A list-valued ``source_concept`` or ``source_measure_id`` still names
    several series (OBR's two universal credit cells), each read across the
    mapped tables. As for the measure map, the years are checked when the
    window resolves.
    """

    if reference.ledger_selector.get("source_table") not in (None, ""):
        raise ValueError(
            f"LedgerTargetReference {reference.name!r}: {operation} declares both "
            f"'source_table' and {SOURCE_TABLE_BY_OPENING_YEAR!r}; the map names "
            "each year's table itself."
        )
    _table_by_opening_year(reference.ledger_selector[SOURCE_TABLE_BY_OPENING_YEAR])


def _calendar_year_window_opening_year(
    fact: object, reference: LedgerTargetReference
) -> int:
    period_type = _str_at(fact, "period", "type")
    if period_type not in CALENDAR_YEAR_WINDOW_PERIOD_TYPES:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: calendar_year_window "
            f"composes fiscal- or tax-year facts; matched a {period_type!r} fact."
        )
    period_key = _period_key(fact)
    if not period_key[0] or period_key[1] % 100 != 99:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: calendar_year_window "
            "matched a fact whose annual period does not parse to an opening year."
        )
    return period_key[1] // 100


def _validate_calendar_year_window_scaled_by_ratio_reference(
    reference: LedgerTargetReference,
) -> None:
    """A window of the reference's own series, then two pinned ratio operands."""

    prefix = (
        f"LedgerTargetReference {reference.name!r}: "
        f"{CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO}"
    )
    roles = [
        str(operand.get("role")) if isinstance(operand, Mapping) else ""
        for operand in reference.value_operands
    ]
    if roles != list(SCALED_BY_RATIO_OPERAND_ROLES):
        raise ValueError(
            f"{prefix} requires exactly ordered "
            f"{'/'.join(SCALED_BY_RATIO_OPERAND_ROLES)} operands, got {roles!r}."
        )
    if {key for key in reference.value_operands[0] if key != "role"}:
        raise ValueError(
            f"{prefix}: the base operand is the reference's own windowed series "
            "and takes no overrides."
        )
    for operand in reference.value_operands[1:]:
        if "period_type" not in operand or "period_value" not in operand:
            raise ValueError(
                f"{prefix}: operand {operand.get('role')!r} must pin its own "
                "period_type and period_value; the window's years are the "
                "base's, not the ratio's."
            )
        count = operand.get("expected_member_count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"{prefix}: operand {operand.get('role')!r} "
                "expected_member_count must be a positive integer."
            )
        if not {key for key in operand} - {
            "period_type",
            "period_value",
            *_WINDOW_RATIO_OPERAND_METADATA_KEYS,
        }:
            raise ValueError(
                f"{prefix}: operand {operand.get('role')!r} needs at least one "
                "selector field besides its period."
            )
        members = _window_ratio_member_keys(operand)
        if (count > 1 or members) and (
            len(members) != 1
            or len({str(value) for value in operand[members[0]]}) != count
            or len(operand[members[0]]) != count
        ):
            raise ValueError(
                f"{prefix}: operand {operand.get('role')!r} sums {count} "
                "cell(s), so it names them in exactly one list-valued selector "
                f"field of {count} distinct member(s), got {members!r}."
            )
    for operand in reference.value_operands[1:]:
        if "complement" in operand and (
            operand.get("role") != "numerator" or operand["complement"] is not True
        ):
            raise ValueError(
                f"{prefix}: only the numerator may declare complement, and only "
                "as true."
            )
    if SOURCE_MEASURE_ID_BY_OPENING_YEAR in reference.ledger_selector:
        raise ValueError(
            f"{prefix} reads one series; {SOURCE_MEASURE_ID_BY_OPENING_YEAR!r} is "
            "refused."
        )
    if SOURCE_TABLE_BY_OPENING_YEAR in reference.ledger_selector:
        _validate_table_by_opening_year(reference, CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO)
    if reference.period_match_policy != "latest_not_after":
        raise ValueError(
            f"{prefix} resolves the two years overlapping the target calendar "
            "year itself; declare period_match_policy 'latest_not_after' (the "
            "default)."
        )
    period_key = _period_key_from_value(reference.period)
    if reference.period is None or not period_key[0] or period_key[1] % 100 != 99:
        raise ValueError(f"{prefix} requires an explicit calendar-year target period.")


#: Keys of a window-ratio operand that are not selector fields. A numerator
#: may declare ``complement``: its cells are taken off the denominator, so
#: the share is the rest of the published universe (England's chargeable
#: dwellings less second homes and empty dwellings, microcosm#1095 D14f).
_WINDOW_RATIO_OPERAND_METADATA_KEYS = frozenset(
    ("role", "expected_member_count", "label", "complement")
)


def _window_ratio_member_keys(operand: Mapping[str, object]) -> list[str]:
    """The operand's list-valued selector fields, which name its members."""

    return sorted(
        str(key)
        for key, value in operand.items()
        if key not in _WINDOW_RATIO_OPERAND_METADATA_KEYS
        and key not in {"dimensions", "dimension_values"}
        and isinstance(value, (list, tuple))
    )


def _window_ratio_operand_selectors(
    reference: LedgerTargetReference,
) -> tuple[Mapping[str, object], ...]:
    """The numerator and denominator selectors, complete as declared."""

    return tuple(
        {
            str(key): value
            for key, value in operand.items()
            if key not in _WINDOW_RATIO_OPERAND_METADATA_KEYS
        }
        for operand in reference.value_operands[1:]
    )


def _window_ratio_numerator_is_complement(reference: LedgerTargetReference) -> bool:
    """Whether the numerator's cells are taken off the denominator."""

    return reference.value_operands[1].get("complement") is True


def _window_ratio_operand_counts(reference: LedgerTargetReference) -> tuple[int, int]:
    """How many facts the numerator and the denominator each sum."""

    numerator, denominator = (
        int(operand.get("expected_member_count", 1))
        for operand in reference.value_operands[1:]
    )
    return numerator, denominator


def _window_ratio_window_facts(
    reference: LedgerTargetReference, facts: tuple[object, ...]
) -> tuple[object, ...]:
    """The window's own facts, without the operands that translate it."""

    return facts[: len(facts) - sum(_window_ratio_operand_counts(reference))]


def _validate_window_ratio_operand_fact(
    fact: object, reference: LedgerTargetReference
) -> None:
    """A ratio operand is checked against the reference's assertion policy."""

    if not _assertion_allowed(reference, fact):
        raise ValueError(
            f"Ledger target reference {reference.name!r} assertion_policy="
            f"{reference.assertion_policy!r} does not allow ratio operand "
            f"assertion {_fact_assertion(fact)!r}."
        )


def _resolve_calendar_year_window_scaled_by_ratio_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
    facts: tuple[object, ...],
) -> tuple[object, ...]:
    """Resolve the window from the selector and each operand exactly once.

    The window's two years come from the reference's own series, as for
    ``calendar_year_window``. The numerator and denominator are complete
    selectors over every fact the compile was given, each pinned at its own
    period, and each must match exactly its declared number of facts (one
    unless ``expected_member_count`` says otherwise), which it sums.
    """

    resolved = list(
        _resolve_calendar_year_window_reference_facts(reference, eligible_matches)
    )
    for operand, selector in zip(
        reference.value_operands[1:],
        _window_ratio_operand_selectors(reference),
        strict=True,
    ):
        matches = [fact for fact in facts if _fact_matches_selector(fact, selector)]
        expected = int(operand.get("expected_member_count", 1))
        if len(matches) != expected:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: "
                f"{CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO} operand "
                f"{operand.get('role')!r} matched {len(matches)} facts; expected "
                f"exactly {expected}."
            )
        # Each named member matches exactly once: two facts for one member
        # and none for another would pass the total alone.
        for key in _window_ratio_member_keys(operand):
            for member in operand[key]:
                hits = sum(
                    1
                    for fact in matches
                    if str(member) in {str(v) for v in _selector_candidates(fact, key)}
                )
                if hits != 1:
                    raise ValueError(
                        f"Ledger target reference {reference.name!r}: "
                        f"{CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO} operand "
                        f"{operand.get('role')!r} member {member!r} of {key!r} "
                        f"matched {hits} facts; expected exactly one."
                    )
        resolved.extend(sorted(matches, key=lambda fact: str(_fact_key(fact))))
    return tuple(resolved)


def _resolve_calendar_year_window_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    """Resolve the two annual facts whose years overlap the target calendar year.

    Publishers who report by April-to-March years give a calendar year Y three
    months of the year opening in Y-1 and nine of the year opening in Y. The
    resolver takes both facts from one series identity (one publication, one
    measure, one cell) and refuses a window with either year absent, so a
    partial window never lands as a value. A selector that maps each year to
    its own measure (``source_measure_id_by_opening_year``) has matched every
    fact to the measure its year declares, so those measures form the one
    series and the measure drops out of its identity. A selector that maps
    each year to its own table (``source_table_by_opening_year``) does the
    same for two editions of one publication: the record set, which belongs
    to its table, drops out of the identity. A selector that names
    several series (a list-valued ``source_concept``, ``source_measure_id`` or
    ``source_table``) resolves both years of every one of them, so the window
    of a sum of series is the sum of the windowed series; a series short of a
    year refuses the whole window.
    """

    target_year = _calendar_year_from_reference(reference)
    wanted = {target_year + offset for offset in CALENDAR_YEAR_WINDOW_WEIGHTS}
    for map_key, parse_map in (
        (SOURCE_MEASURE_ID_BY_OPENING_YEAR, _measure_id_by_opening_year),
        (SOURCE_TABLE_BY_OPENING_YEAR, _table_by_opening_year),
    ):
        year_map = reference.ledger_selector.get(map_key)
        if year_map is None:
            continue
        declared = sorted(parse_map(year_map))
        if declared != sorted(wanted):
            raise ValueError(
                f"Ledger target reference {reference.name!r}: calendar_year_window "
                f"of calendar year {target_year} reads the years opening in "
                f"{sorted(wanted)}; {map_key!r} maps {declared}."
            )
    expected_series = _calendar_year_window_series_count(reference)
    partitions: dict[tuple[str, ...], dict[int, list[object]]] = {}
    for fact in eligible_matches:
        opening_year = _calendar_year_window_opening_year(fact, reference)
        if opening_year not in wanted:
            continue
        series_key = _selector_period_invariant_key(
            fact,
            include_measure=(
                SOURCE_MEASURE_ID_BY_OPENING_YEAR not in reference.ledger_selector
            ),
            include_record_set=(
                SOURCE_TABLE_BY_OPENING_YEAR not in reference.ledger_selector
            ),
        )
        partitions.setdefault(series_key, {}).setdefault(opening_year, []).append(fact)
    complete = {key: years for key, years in partitions.items() if set(years) == wanted}
    if not complete or len(complete) != expected_series:
        present = sorted({year for years in partitions.values() for year in years})
        series_phrase = (
            "one series"
            if expected_series == 1
            else f"each of {expected_series} series (one per declared selector member)"
        )
        if not complete:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: calendar_year_window "
                f"needs the years opening in {sorted(wanted)} from {series_phrase}; "
                f"found {present or 'none'}."
            )
        if expected_series == 1:
            raise ValueError(
                f"Ledger target reference {reference.name!r}: calendar_year_window "
                f"matched {len(complete)} series identities carrying both years; "
                "narrow the selector to one."
            )
        raise ValueError(
            f"Ledger target reference {reference.name!r}: calendar_year_window "
            f"needs both years opening in {sorted(wanted)} from {series_phrase}; "
            f"{len(complete)} of {len(partitions)} matched series carry both."
        )
    if expected_series > 1:
        # Each declared member carries exactly one series: two series of one
        # member and none of another would pass the count alone (the review
        # of microcosm#1422), as the ratio operands' members are checked.
        members: dict[tuple[str, ...], int] = {}
        for years in complete.values():
            member = _calendar_year_window_member(
                next(iter(years.values()))[0], reference
            )
            members[member] = members.get(member, 0) + 1
        if len(members) != expected_series or any(
            count != 1 for count in members.values()
        ):
            found = ", ".join(
                f"{'/'.join(member)} x{count}"
                for member, count in sorted(members.items())
            )
            raise ValueError(
                f"Ledger target reference {reference.name!r}: calendar_year_window "
                f"needs one series per declared selector member ({expected_series}); "
                f"the complete series stand for {found}."
            )
    resolved: list[object] = []
    for key in sorted(complete):
        years = complete[key]
        for year in sorted(years):
            members = years[year]
            if len(members) != 1:
                raise ValueError(
                    f"Ledger target reference {reference.name!r}: calendar_year_window "
                    f"matched {len(members)} facts for the year opening in {year}; "
                    "expected exactly one per series."
                )
            resolved.append(members[0])
    return tuple(resolved)


_CALENDAR_YEAR_WINDOW_SERIES_KEYS = (
    "source_concept",
    "source_measure_id",
    "source_table",
)


def _calendar_year_window_series_count(reference: LedgerTargetReference) -> int:
    """How many series the window composes: one per declared selector member.

    A dimension the selector pins to several values names one series per value,
    as a list-valued source concept, measure or table does (DfI's Translink
    receipts by service).
    """

    selector = reference.ledger_selector or {}
    count = 1
    for key in _CALENDAR_YEAR_WINDOW_SERIES_KEYS:
        value = selector.get(key)
        if isinstance(value, (list, tuple)):
            count *= len(value)
    for name in _calendar_year_window_listed_dimensions(reference):
        count *= len(selector["dimension_values"][name])
    return count


def _calendar_year_window_member(
    fact: object, reference: LedgerTargetReference
) -> tuple[str, ...]:
    """The declared selector member a series stands for: its value of each
    list-valued source key and of each dimension pinned to several values."""

    selector = reference.ledger_selector or {}
    member: list[str] = []
    for key in _CALENDAR_YEAR_WINDOW_SERIES_KEYS:
        declared = selector.get(key)
        if isinstance(declared, (list, tuple)):
            candidates = {str(value) for value in _selector_candidates(fact, key)}
            member.append(
                ",".join(sorted(str(v) for v in declared if str(v) in candidates))
            )
    dimensions = _dimensions(fact)
    for name in _calendar_year_window_listed_dimensions(reference):
        member.append(f"{name}={dimensions.get(name, '')}")
    return tuple(member)


def _calendar_year_window_listed_dimensions(
    reference: LedgerTargetReference,
) -> tuple[str, ...]:
    """The dimensions the selector pins to several values, in name order."""

    pins = (reference.ledger_selector or {}).get("dimension_values") or {}
    return tuple(
        sorted(
            str(name)
            for name, value in pins.items()
            if isinstance(value, (list, tuple)) and len(value) > 1
        )
    )


def _calendar_year_window_series_label(
    fact: object, reference: LedgerTargetReference
) -> str:
    label = (
        _str_at(fact, "observed_measure", "source_measure_id")
        or _str_at(fact, "layout", "measure_id")
        or _source_measure_concept(fact)
    )
    listed = _calendar_year_window_listed_dimensions(reference)
    if not listed:
        return label
    dimensions = _dimensions(fact)
    members = ",".join(f"{name}={dimensions.get(name, '')}" for name in listed)
    return f"{label}[{members}]"


def _calendar_year_window_weights(
    reference: LedgerTargetReference, facts: tuple[object, ...]
) -> tuple[float, ...]:
    target_year = _calendar_year_from_reference(reference)
    return tuple(
        CALENDAR_YEAR_WINDOW_WEIGHTS[
            _calendar_year_window_opening_year(fact, reference) - target_year
        ]
        for fact in facts
    )


def _calendar_year_window_metadata(
    reference: LedgerTargetReference, facts: tuple[object, ...]
) -> dict[str, str]:
    target_year = _calendar_year_from_reference(reference)
    series_count = _calendar_year_window_series_count(reference)
    members: dict[str, dict[str, str]] = {}
    for fact in facts:
        opening_year = _calendar_year_window_opening_year(fact, reference)
        member_key = (
            str(opening_year)
            if series_count == 1
            else f"{_calendar_year_window_series_label(fact, reference)}@{opening_year}"
        )
        members[member_key] = {
            "weight": f"{CALENDAR_YEAR_WINDOW_WEIGHTS[opening_year - target_year]:.15g}",
            "value": f"{_numeric_fact_value(fact, reference):.15g}",
            "assertion": _fact_assertion(fact),
            "fact_key": _fact_key(fact) or _source_record_id(fact),
        }
    return {
        "ledger_calendar_year_window_target_year": str(target_year),
        "ledger_calendar_year_window_series": str(series_count),
        "ledger_calendar_year_window_members": json.dumps(
            members, sort_keys=True, separators=(",", ":")
        ),
    }


def _resolve_latest_plateau_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    monthly_matches = _monthly_operation_matches(
        reference, eligible_matches, operation="latest_plateau"
    )
    partition = _latest_series_partition(
        reference, monthly_matches, operation="latest_plateau"
    )
    sorted_partition = tuple(sorted(partition, key=_fact_month_sort_key))
    plateau_value = _numeric_fact_value(sorted_partition[-1], reference)
    plateau = [sorted_partition[-1]]
    expected_month = _fact_month_index(sorted_partition[-1]) - 1
    for fact in reversed(sorted_partition[:-1]):
        if _fact_month_index(fact) != expected_month:
            break
        if _numeric_fact_value(fact, reference) != plateau_value:
            break
        plateau.append(fact)
        expected_month -= 1
    return tuple(reversed(plateau))


def _resolve_count_x_mean_reference_facts(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
) -> tuple[object, ...]:
    partitions: dict[
        tuple[tuple[str, ...], tuple[int, int, str]], dict[str, list[object]]
    ] = {}
    for fact in eligible_matches:
        role = _count_mean_fact_role(fact)
        if not role:
            continue
        key = (
            _selector_count_mean_partition_key(fact),
            _reference_period_partition_key(fact, reference),
        )
        partitions.setdefault(key, {}).setdefault(role, []).append(fact)

    valid = {
        key: roles
        for key, roles in partitions.items()
        if len(roles.get("count", ())) == 1 and len(roles.get("mean", ())) == 1
    }
    if not valid:
        observed = {
            key: {role: len(facts) for role, facts in roles.items()}
            for key, roles in partitions.items()
        }
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'count_x_mean' requires exactly one count fact and one mean fact "
            f"in a shared selector partition; observed {observed!r}."
        )
    latest_period = max(period_key for _, period_key in valid)
    latest = [
        roles for (_, period_key), roles in valid.items() if period_key == latest_period
    ]
    if len(latest) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'count_x_mean' matched multiple count/mean selector partitions "
            "at the latest eligible period."
        )
    roles = latest[0]
    return (roles["count"][0], roles["mean"][0])


def _monthly_operation_matches(
    reference: LedgerTargetReference,
    eligible_matches: list[object],
    *,
    operation: str,
) -> list[object]:
    non_month = [
        _at(fact, "period", "value")
        for fact in eligible_matches
        if _str_at(fact, "period", "type") != "month"
    ]
    if non_month:
        examples = tuple(str(period) for period in non_month[:5])
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            f"{operation!r} requires monthly Ledger facts; non-month period "
            f"values include {examples!r}."
        )
    return eligible_matches


def _calendar_year_from_reference(reference: LedgerTargetReference) -> int:
    period_key = _period_key_from_value(reference.period)
    if not period_key[0]:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            "'calendar_year_average' requires a parseable target period."
        )
    return period_key[1] // 100


def _latest_series_partition(
    reference: LedgerTargetReference,
    matches: list[object],
    *,
    operation: str,
) -> tuple[object, ...]:
    partitions: dict[tuple[str, ...], list[object]] = {}
    for fact in matches:
        partitions.setdefault(_selector_period_invariant_key(fact), []).append(fact)
    if not partitions:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            f"{operation!r} did not match a Ledger fact partition."
        )
    latest_period = max(_period_key(fact) for fact in matches)
    latest_partitions = [
        partition
        for partition in partitions.values()
        if max(_period_key(fact) for fact in partition) == latest_period
    ]
    if len(latest_partitions) != 1:
        raise ValueError(
            f"Ledger target reference {reference.name!r}: value_operation="
            f"{operation!r} matched multiple monthly Ledger series identities "
            "at the latest eligible period."
        )
    return tuple(sorted(latest_partitions[0], key=_fact_month_sort_key))


def _fact_month_sort_key(fact: object) -> tuple[int, str]:
    return (_fact_month_index(fact), _fact_key(fact) or _source_record_id(fact) or "")


def _fact_month_index(fact: object) -> int:
    year, month = _fact_year_month(fact)
    return year * 12 + month


def _fact_year_month(fact: object) -> tuple[int, int]:
    period_key = _period_key(fact)
    if not period_key[0]:
        raise ValueError("Ledger monthly period is not parseable.")
    year_month = period_key[1]
    year = year_month // 100
    month = year_month % 100
    if month < 1 or month > 12:
        raise ValueError("Ledger monthly period does not carry a month value.")
    return year, month


def _selector_sum_partition_key(fact: object) -> tuple[str, ...]:
    invariant = _selector_period_invariant_key(fact)
    return invariant[:8] + invariant[11:]


def _selector_count_mean_partition_key(fact: object) -> tuple[str, ...]:
    return (
        _source_name(fact),
        _str_at(fact, "geography", "level"),
        _str_at(fact, "geography", "id"),
        _str_at(fact, "entity", "name"),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_id")),
        _str_at(fact, "layout", "record_set_spec_id"),
        _str_at(fact, "layout", "groupby_dimension"),
        _normalized_period_bearing_id(_str_at(fact, "layout", "groupby_value_id")),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
        _domain(fact),
    )


def _count_mean_fact_role(fact: object) -> str:
    measure_id = (
        _str_at(fact, "observed_measure", "source_measure_id")
        or _str_at(fact, "layout", "measure_id")
    ).lower()
    if measure_id.endswith("_count") or measure_id == "count":
        return "count"
    if measure_id.endswith("_mean") or measure_id == "mean":
        return "mean"
    return ""


def restamp_reference_period(
    reference: LedgerTargetReference, period: int | str
) -> LedgerTargetReference:
    """The reference measured at another target period.

    A declared ``observation_period`` keeps its offset from the target period:
    a row that reads DfE's January census after its model year reads the
    census after the new one. Restamping only ``period`` left every such row
    on the census it was written for (the review of microcosm#1422).
    """

    if reference.observation_period is None:
        return replace(reference, period=period)
    if reference.period is None:
        raise ValueError(
            f"LedgerTargetReference {reference.name!r}: an observation_period "
            "needs a target period to keep its offset from."
        )
    shift = _period_year(period) - _period_year(reference.period)
    return replace(
        reference,
        period=period,
        observation_period=_shift_period_years(reference.observation_period, shift),
    )


def _period_year(period: int | str) -> int:
    parsed, ordinal, _ = _period_key_from_value(period)
    if not parsed:
        raise ValueError(f"cannot read a year from the period {period!r}.")
    return ordinal // 100


def _shift_period_years(period: int | str, years: int) -> int | str:
    """The period ``years`` later, in its own spelling.

    The spelling is classified as the ledger's own period parser reads it
    (``_normalize_period_value``): an annual range (``"2024-25"``,
    ``"2024-2025"``, ``"fy2024_25"``) moves both ends, and a year
    (``2026``, ``"FY2026"``) or a month (``"2026-01"``, ``"2011-12"``,
    ``"2026-1"``) moves its year. A spelling the parser does not read
    refuses, unless it does not move (the third review of microcosm#1422).
    """

    if years == 0:
        return period
    if isinstance(period, int):
        return period + years
    text = str(period).strip()
    (parsed, _, _), _, range_end, _ = _normalize_period_value(text)
    if parsed and range_end is not None:
        ranged = re.fullmatch(r"(\D*)(\d{4})([-_])(\d{2}|\d{4})", text)
        if ranged:
            start = int(ranged.group(2)) + years
            end = start + 1 if len(ranged.group(4)) == 4 else (start + 1) % 100
            width = len(ranged.group(4))
            return f"{ranged.group(1)}{start}{ranged.group(3)}{end:0{width}d}"
    elif parsed:
        dated = re.fullmatch(r"(\D*)(\d{4})((?:[-_]\d{1,2})?)", text)
        if dated:
            return f"{dated.group(1)}{int(dated.group(2)) + years}{dated.group(3)}"
    raise ValueError(f"cannot move the period {period!r} by {years} year(s).")


def _selection_period(reference: LedgerTargetReference) -> int | str | None:
    """The period a reference selects its facts at: its observation period if
    it declares one, else its target period."""

    if reference.observation_period is not None:
        return reference.observation_period
    return reference.period


def _selection_period_label(reference: LedgerTargetReference) -> str:
    if reference.observation_period is not None:
        return "observation period"
    return "target period"


def _eligible_selector_matches(
    reference: LedgerTargetReference,
    matches: list[object],
) -> list[object]:
    target_period_key = _period_key_from_value(_selection_period(reference))
    if reference.period_match_policy == "source_window":
        return [
            fact
            for fact in matches
            if _fact_matches_selector(fact, reference.ledger_selector)
            and _assertion_allowed(reference, fact)
        ]
    if reference.period_match_policy == "exact":
        return [
            fact
            for fact in matches
            if _exact_period_matches(fact, reference)
            and _assertion_allowed(reference, fact)
        ]
    return [
        fact
        for fact in matches
        if _not_after_target_period(_period_key(fact), target_period_key)
        and _assertion_allowed(reference, fact)
    ]


def _assertion_allowed(reference: LedgerTargetReference, fact: object) -> bool:
    assertion = _fact_assertion(fact)
    if assertion == "observation":
        return True
    if assertion == "source_projection":
        return reference.assertion_policy == "allow_source_projection"
    return False


def _validate_resolved_reference_fact(
    fact: object,
    reference: LedgerTargetReference,
) -> None:
    """Apply consumer policy after either identifier or selector resolution.

    Selector filtering still uses these predicates to choose an eligible row
    from a series.  This post-resolution check is the authority: exact keys and
    source-record identifiers must not bypass assertion, period, or declared
    geography-vintage policy.
    """

    _validate_reference_period(fact, reference)
    if not _assertion_allowed(reference, fact):
        raise ValueError(
            f"Ledger target reference {reference.name!r} assertion_policy="
            f"{reference.assertion_policy!r} does not allow resolved fact "
            f"assertion {_fact_assertion(fact)!r}."
        )
    vintage_pin = reference.ledger_selector.get("geography_vintage")
    if vintage_pin is not None and vintage_pin != "":
        if not _str_at(fact, "geography", "vintage") or not _fact_matches_selector(
            fact, {"geography_vintage": vintage_pin}
        ):
            raise ValueError(
                f"Ledger target reference {reference.name!r} requires geography "
                f"vintage {vintage_pin!r}, but resolved fact has vintage "
                f"{_at(fact, 'geography', 'vintage')!r}."
            )
    entity_pin = reference.ledger_selector.get("entity_name")
    if entity_pin is not None and _str_at(fact, "entity", "name") != entity_pin:
        raise ValueError(
            f"Ledger target reference {reference.name!r} requires entity_name "
            f"{entity_pin!r}, but resolved fact has entity "
            f"{_at(fact, 'entity', 'name')!r}."
        )


def _validate_reference_period(fact: object, reference: LedgerTargetReference) -> None:
    """Enforce the declared period consumer contract after either resolution path.

    An exact-period reference may consume either an observation at that period
    or a Ledger ``source_projection`` whose published fact period is that same
    target period. It may not silently substitute an older observation. The
    projection's source period belongs in Ledger lineage; the consumer still
    binds the projected fact to the period it estimates.
    """

    if reference.period_match_policy == "source_window":
        if _str_at(fact, "period", "type") != "month" or _str_at(
            fact, "period", "value"
        ) not in _declared_source_months(reference):
            raise ValueError("Resolved fact lies outside the declared monthly window.")
        return
    if reference.period_match_policy != "exact":
        if not _not_after_target_period(
            _period_key(fact), _period_key_from_value(_selection_period(reference))
        ):
            raise ValueError(
                f"Ledger target reference {reference.name!r} requires a fact "
                f"at or before {_selection_period_label(reference)} "
                f"{_selection_period(reference)!r}, but resolved "
                f"fact period {_at(fact, 'period', 'value')!r}."
            )
        return
    if not _exact_period_matches(fact, reference):
        raise ValueError(
            f"Ledger target reference {reference.name!r} requires exact period "
            f"{_selection_period(reference)!r}, but resolved fact period "
            f"{_at(fact, 'period', 'type')!r}:"
            f"{_at(fact, 'period', 'value')!r}. A period mismatch must resolve "
            "through a Ledger source_projection at the target period, never "
            "through a silently stale observation."
        )


def _fact_assertion(fact: object) -> str:
    return _str_at(fact, "assertion") or "observation"


def _latest_period_selector_match(
    reference: LedgerTargetReference,
    matches: list[object],
) -> object | None:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    semantic_keys = {_selector_period_invariant_key(fact) for fact in matches}
    if len(semantic_keys) != 1:
        return None

    target_period_key = _period_key_from_value(_selection_period(reference))
    best: tuple[int, int, str] | None = None
    best_matches: list[object] = []
    for fact in matches:
        period_key = _period_key(fact)
        if best is None or _prefer_period_candidate(
            period_key,
            best,
            target_period_key=target_period_key,
        ):
            best = period_key
            best_matches = [fact]
        elif period_key == best:
            best_matches.append(fact)
    if len(best_matches) == 1:
        return best_matches[0]
    return None


def _selector_period_invariant_key(
    fact: object, *, include_measure: bool = True, include_record_set: bool = True
) -> tuple[str, ...]:
    return (
        _source_name(fact),
        (
            _str_at(fact, "observed_measure", "source_measure_id")
            or _str_at(fact, "layout", "measure_id")
        )
        if include_measure
        else "",
        _source_measure_concept(fact),
        _str_at(fact, "geography", "level"),
        _str_at(fact, "geography", "id"),
        _str_at(fact, "entity", "name"),
        _str_at(fact, "aggregation", "method"),
        _normalized_record_set_id(_str_at(fact, "layout", "record_set_id"))
        if include_record_set
        else "",
        _str_at(fact, "layout", "groupby_dimension"),
        _normalized_period_bearing_id(_str_at(fact, "layout", "groupby_value_id")),
        json.dumps(_dimensions(fact), sort_keys=True, separators=(",", ":")),
        json.dumps(_constraint_rows(fact), sort_keys=True, separators=(",", ":")),
        _domain(fact),
    )


def _normalized_record_set_id(record_set_id: str) -> str:
    if not record_set_id:
        return ""
    return ".".join(
        normalized
        for part in record_set_id.split(".")
        if (normalized := _normalized_record_set_part(part))
    )


def _normalized_record_set_part(value: str) -> str:
    if _is_period_token(value):
        return ""
    normalized = value.lower().replace("-", "_")
    pieces = [
        _strip_trailing_vintage_year(piece)
        for piece in normalized.split("_")
        if piece and not _is_period_fragment(piece)
    ]
    return "_".join(pieces)


_TRAILING_VINTAGE_YEAR = re.compile(r"^([a-z]{2,})((?:19|20)[0-9]{2})$")
#: Word-plus-year tokens that name a classification revision, not a publication
#: vintage: they must keep their year, or two classifications would read as one
#: series. Grow this set when a new one enters a record-set id.
_CLASSIFICATION_REVISION_TOKENS = frozenset({"sic2007"})


def _strip_trailing_vintage_year(piece: str) -> str:
    """``ctaxbase2025`` -> ``ctaxbase``: a vintage year glued to a word.

    The strip is generic: it also folds ``ons.mid2023…`` and ``ons.mid2024…``
    (the road-fuel anchors' mid-year estimate vintages) into one series, which
    is intended, since successive MYE vintages are one publication series. A
    classification revision (``sic2007``) is not a vintage and is kept.

    Chronicle names one package per publication year and spells the year
    into the record-set id without a separator (``scotgov.ctaxbase2025.…``,
    ``mhclg.ctb2025.…``). The series-invariant key must read those the way it
    already reads ``fy2025`` and ``september2025``, or every vintage of one
    series looks like a different series and the latest-not-after resolution
    refuses the reference as ambiguous.
    """

    if piece in _CLASSIFICATION_REVISION_TOKENS:
        return piece
    match = _TRAILING_VINTAGE_YEAR.match(piece)
    return match.group(1) if match else piece


#: A numeric month beside its year (``03_2025``, ``2025-03``) in a layout value:
#: ONS's June 2026 public sector employment edition keys its quarter rows so,
#: where the March 2026 edition wrote ``dec2024`` (microcosm#1123).
_NUMERIC_MONTH_YEAR = re.compile(
    r"^(?:(?:0[1-9]|1[0-2])[_-]?(?:19|20)[0-9]{2}|(?:19|20)[0-9]{2}[_-]?(?:0[1-9]|1[0-2]))$"
)


def _normalized_period_bearing_id(value: str) -> str:
    if _is_period_token(value) or _NUMERIC_MONTH_YEAR.match(value.strip().lower()):
        return ""
    return _normalized_record_set_part(value)


def _is_period_fragment(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    if _is_period_token(normalized):
        return True
    if normalized.startswith("year") and _is_period_token(normalized[4:]):
        return True
    month_names = (
        "jan",
        "january",
        "feb",
        "february",
        "mar",
        "march",
        "apr",
        "april",
        "may",
        "jun",
        "june",
        "jul",
        "july",
        "aug",
        "august",
        "sep",
        "sept",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
    )
    return any(
        normalized.startswith(month)
        and normalized[len(month) :].isdigit()
        and len(normalized[len(month) :]) == 4
        for month in month_names
    )


def _is_period_token(value: str) -> bool:
    period_key = _period_key_from_value(value)
    # Source-table numbers such as table_1_2 are identity, not year tokens.
    return bool(period_key[0]) and 1000 <= period_key[1] // 100 <= 9999


def _comparable_period_value(fact: object) -> object:
    """The period value the surface compares a fact on.

    Every fiscal-year label on the surface is read as the opening year
    (OBR's and HMRC's ``2025`` is FY2025-26). DfT labels its reporting year
    by the March end year (``2025`` is YE March 2025, FY2024-25) and says so
    in the fact's own coverage dates, so a fiscal-year fact that carries
    ``period_coverage.start_date`` is keyed on that start year. Facts without
    coverage keep their label; the publisher's label is recorded beside the
    comparable value (``ledger_fact_period_label``) whenever they differ.
    """

    value = _at(fact, "period", "value")
    if _str_at(fact, "period", "type") != "fiscal_year":
        return value
    start = _str_at(fact, "period_coverage", "start_date")
    if len(start) < 4 or not start[:4].isdigit():
        return value
    label = str(value).strip()
    if not label.isdigit():
        return value
    return int(start[:4])


def _period_key(fact: object) -> tuple[int, int, str]:
    return _period_key_from_value(_comparable_period_value(fact))


def _exact_period_matches(fact: object, reference: LedgerTargetReference) -> bool:
    """Match exact periods by semantic value while retaining period-kind pins."""

    expected_value = _selection_period(reference)
    actual_value = _comparable_period_value(fact)
    actual_type = _str_at(fact, "period", "type")
    selector_type = str(reference.ledger_selector.get("period_type", ""))
    expected_type_hint = period_type_hint(expected_value)
    if selector_type and expected_type_hint and selector_type != expected_type_hint:
        return False
    expected_type = selector_type or expected_type_hint
    if expected_type and actual_type != expected_type:
        return False
    actual_type_hint = period_type_hint(actual_value)
    if actual_type_hint and actual_type_hint != actual_type:
        return False
    return period_values_semantically_equal(
        expected_value,
        actual_value,
        declared_type=expected_type or actual_type,
    )


def _reference_period_partition_key(
    fact: object,
    reference: LedgerTargetReference,
) -> tuple[int, int, str]:
    period_key = (
        _normalize_period_value(
            _comparable_period_value(fact),
            declared_type=_str_at(fact, "period", "type"),
        )[0]
        if reference.period_match_policy == "exact"
        else _period_key(fact)
    )
    if reference.period_match_policy == "exact" and period_key[0]:
        return (period_key[0], period_key[1], "")
    return period_key


def period_values_semantically_equal(
    left: object, right: object, *, declared_type: str = ""
) -> bool:
    """Compare valid period spellings, using the declared type for untyped values.

    Opaque publisher labels retain literal equality. Malformed numeric period
    shapes do not, even when the same invalid spelling appears on both sides.
    """

    left_key, _, left_range_end, left_malformed = _normalize_period_value(
        left, declared_type=declared_type
    )
    right_key, _, right_range_end, right_malformed = _normalize_period_value(
        right, declared_type=declared_type
    )
    if left_malformed or right_malformed:
        return False
    if left_key[0] and right_key[0]:
        return left_key[:2] == right_key[:2] and left_range_end == right_range_end
    return left_key == right_key


def period_type_hint(value: object) -> str:
    """Return the period-kind prefix recognized by the shared normalizer."""

    return _normalize_period_value(value)[1]


def _period_key_from_value(value: object) -> tuple[int, int, str]:
    return _normalize_period_value(value)[0]


def _normalize_period_value(
    value: object,
    *,
    declared_type: str = "",
) -> tuple[tuple[int, int, str], str, int | None, bool]:
    """Normalize one scalar/typed period spelling once for all consumers.

    Annual aliases (``ty``, ``cy``, ``fy``, and ``ay``) and their long forms
    share one parser.  A two-part value under an annual prefix is an annual
    range (for example ``academic_year_2023_24``). A declared annual type also
    disambiguates untyped ``2003_04`` as a range rather than a monthly point.
    Annual ranges must end in the following year. Keep that explicit end year
    in the semantic identity: a range is not interchangeable with a scalar year.
    The final flag distinguishes malformed numeric shapes from opaque labels,
    so only the latter can fall back to literal equality.
    """

    label = "" if value is None else str(value)
    normalized = label.strip().lower().replace("-", "_")
    type_hint = ""
    long_prefixes = (
        ("tax_year", "tax_year"),
        ("calendar_year", "calendar_year"),
        ("fiscal_year", "fiscal_year"),
        ("academic_year", "academic_year"),
        ("month", "month"),
    )
    for prefix, period_type in long_prefixes:
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :].lstrip("_")
            if suffix[:1].isdigit():
                normalized = suffix
                type_hint = period_type
            break
    if not type_hint:
        aliases = {
            "ty": "tax_year",
            "cy": "calendar_year",
            "fy": "fiscal_year",
            "ay": "academic_year",
        }
        alias = normalized[:2]
        suffix = normalized[2:].lstrip("_")
        if alias in aliases and suffix[:1].isdigit():
            normalized = suffix
            type_hint = aliases[alias]

    annual_types = {"tax_year", "calendar_year", "fiscal_year", "academic_year"}
    if (
        not type_hint
        and declared_type in annual_types | {"month"}
        and normalized[:1].isdigit()
    ):
        type_hint = declared_type

    parts = normalized.split("_", maxsplit=1)
    if len(parts) == 2:
        year, suffix = parts
        if year.isdigit() and suffix.isdigit() and len(suffix) in {1, 2, 4}:
            annual_range = type_hint in annual_types or (
                not type_hint and (len(suffix) == 4 or int(suffix) > 12)
            )
            if annual_range:
                range_end = int(year) + 1
                expected_suffix = range_end if len(suffix) == 4 else range_end % 100
                if len(suffix) not in {2, 4} or int(suffix) != expected_suffix:
                    return (0, 0, label), type_hint, None, True
                return (1, int(year) * 100 + 99, label), type_hint, range_end, False
            if len(suffix) <= 2 and 1 <= int(suffix) <= 12:
                return (
                    (1, int(year) * 100 + int(suffix), label),
                    type_hint,
                    None,
                    False,
                )
        numeric_shape = year.isdigit() and all(
            part.isdigit() or not part for part in suffix.split("_")
        )
        return (0, 0, label), type_hint, None, bool(type_hint) or numeric_shape
    if type_hint == "month":
        return (0, 0, label), type_hint, None, True
    try:
        return (1, int(normalized) * 100 + 99, label), type_hint, None, False
    except ValueError:
        return (0, 0, label), type_hint, None, bool(type_hint)


def _prefer_period_candidate(
    candidate: tuple[int, int, str],
    current: tuple[int, int, str],
    *,
    target_period_key: tuple[int, int, str],
) -> bool:
    candidate_is_eligible = _not_after_target_period(candidate, target_period_key)
    current_is_eligible = _not_after_target_period(current, target_period_key)
    if candidate_is_eligible != current_is_eligible:
        return candidate_is_eligible
    return candidate > current


def _not_after_target_period(
    source_period_key: tuple[int, int, str],
    target_period_key: tuple[int, int, str],
) -> bool:
    if not source_period_key[0]:
        return False
    if not target_period_key[0]:
        return True
    return source_period_key[1] <= target_period_key[1]


def _reference_metadata(reference: LedgerTargetReference) -> dict[str, str]:
    metadata = dict(reference.metadata)
    metadata["ledger_value_operation"] = reference.value_operation
    metadata["ledger_assertion_policy"] = reference.assertion_policy
    metadata["ledger_period_match_policy"] = reference.period_match_policy
    if reference.observation_period is not None:
        metadata["ledger_observation_period"] = str(reference.observation_period)
    if reference.value_operation in MONTHLY_WINDOW_OPERATIONS:
        months = _declared_source_months(reference)
        metadata["ledger_source_months"] = json.dumps(months, separators=(",", ":"))
        metadata["ledger_source_month_count"] = str(len(months))
        metadata["ledger_source_cell_count_per_month"] = str(
            len(reference.value_operands) or 1
        )
        metadata["ledger_value_formula"] = (
            "sum_of_declared_cells / declared_month_count"
        )
    if reference.value_operation == "difference":
        metadata["ledger_value_formula"] = "minuend - subtrahend"
    if reference.value_operation == "scaled_by_ratio":
        metadata["ledger_value_formula"] = "base * numerator / denominator"
    if reference.value_operation == ALLOCATED_RESIDUAL_BY_RATIO:
        metadata["ledger_value_formula"] = (
            "(base + (total - partition) * recipient_base / recipients) * growth"
        )
    if reference.value_operation == CALENDAR_YEAR_WINDOW_SCALED_BY_RATIO:
        target_year = _calendar_year_from_reference(reference)
        metadata["ledger_value_formula"] = (
            f"(3/12 * FY{target_year - 1} + 9/12 * FY{target_year}) "
            "* numerator / denominator"
        )
    if reference.value_operation == "linear_combination":
        metadata["ledger_value_formula"] = _linear_combination_formula(reference)
    if reference.value_operation == "calendar_year_window":
        target_year = _calendar_year_from_reference(reference)
        window = (
            f"3/12 * FY{target_year - 1} + 9/12 * FY{target_year} "
            f"(the months to the end of calendar year {target_year})"
        )
        series_count = _calendar_year_window_series_count(reference)
        metadata["ledger_value_formula"] = (
            window
            if series_count == 1
            else f"sum over {series_count} series of [{window}]"
        )
    if reference.metadata.get("composed_from_level"):
        # A composed row's Ledger geography is the row's own, not the
        # representative member fact's (which sits one grain below).
        geography_id = str(reference.metadata.get("geography_id") or "")
        metadata["ledger_geography_level"] = str(
            reference.metadata.get("geography_level") or ""
        )
        metadata["ledger_geography_id"] = geography_id
        metadata["ledger_geography_name"] = _geography_fallback_label(geography_id)
    for key, value in sorted(reference.ledger_selector.items()):
        if isinstance(value, Mapping):
            continue
        if value is not None and value != "":
            metadata[f"ledger_selector_{key}"] = str(value)
    if reference.uprating_index is not None:
        metadata["uprating_index"] = str(reference.uprating_index)
    if reference.uprating_from_period is not None:
        metadata["uprating_from_period"] = str(reference.uprating_from_period)
    if reference.uprating_to_period is not None:
        metadata["uprating_to_period"] = str(reference.uprating_to_period)
    return metadata


def _at(obj: object, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _jsonl_rows(lines: Iterable[str]) -> Iterable[object]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid Ledger JSONL row {line_number}: {exc.msg}"
            ) from exc


def _str_at(obj: object, *path: str) -> str:
    value = _at(obj, *path)
    if value is None:
        return ""
    return str(value)


def _fact_key(fact: object) -> str:
    return (
        _str_at(fact, "aggregate_fact_key")
        or _str_at(fact, "fact_key")
        or _str_at(fact, "legacy_fact_key")
    )


def _source_record_id(fact: object) -> str:
    return _str_at(fact, "source_record_id") or _str_at(
        fact, "lineage", "source_record_id"
    )


def _fact_matches_selector(fact: object, selector: Mapping[str, object]) -> bool:
    """Return whether a consumer fact satisfies a structured reference selector.

    A list-valued scalar key matches by membership; an empty list is refused
    rather than treated as match-nothing, because it reads like match-anything.
    Note ``dimensions: []`` is NOT an empty membership list — it is the
    list-form dimensions selector's exact name-set match for the dimensionless
    total row (see :func:`_dimensions_match`).
    """

    for key, expected in selector.items():
        if key == "dimensions":
            if not _dimensions_match(fact, expected):
                return False
            continue
        if key == "dimension_values":
            if not _dimension_values_match(fact, expected):
                return False
            continue
        if key == SOURCE_MEASURE_ID_BY_OPENING_YEAR:
            if not _measure_by_opening_year_matches(fact, expected):
                return False
            continue
        if key == SOURCE_TABLE_BY_OPENING_YEAR:
            if not _table_by_opening_year_matches(fact, expected):
                return False
            continue
        if expected is None or expected == "":
            continue
        candidates = _selector_candidates(fact, str(key))
        if isinstance(expected, (list, tuple)):
            if not expected:
                raise ValueError(
                    f"Ledger fact selector field {key!r} is an empty list: an "
                    "empty membership list matches nothing while reading like "
                    "match-anything. Pin at least one value or drop the field."
                )
            if not any(str(item) in candidates for item in expected):
                return False
            continue
        if str(expected) not in candidates:
            return False
    return True


def fact_matches_selector(fact: object, selector: Mapping[str, object]) -> bool:
    """Public name for :func:`_fact_matches_selector` (used by fact vendoring)."""

    return _fact_matches_selector(fact, selector)


def selector_field_is_supported(key: str) -> bool:
    """Whether ``key`` is in the closed selector vocabulary."""

    if key in {
        "dimensions",
        "dimension_values",
        SOURCE_MEASURE_ID_BY_OPENING_YEAR,
        SOURCE_TABLE_BY_OPENING_YEAR,
    }:
        return True
    try:
        _selector_candidates({}, key)
    except ValueError:
        return False
    return True


def _selector_candidates(fact: object, key: str) -> tuple[str, ...]:
    if key == "aggregate_fact_key":
        return (_str_at(fact, "aggregate_fact_key"),)
    if key == "semantic_fact_key":
        return (_str_at(fact, "semantic_fact_key"),)
    if key == "legacy_fact_key":
        return (_str_at(fact, "legacy_fact_key"),)
    if key == "source_record_id":
        return (_source_record_id(fact),)
    if key == "source_name":
        return (_source_name(fact),)
    if key == "source_table":
        return (
            _str_at(fact, "source", "source_table"),
            _str_at(fact, "observed_measure", "source_table"),
        )
    if key == "source_measure_id":
        return (
            _str_at(fact, "observed_measure", "source_measure_id"),
            _str_at(fact, "layout", "measure_id"),
        )
    if key == "source_concept":
        return (
            _source_measure_concept(fact),
            _primary_measure_concept(fact),
        )
    if key == "period_type":
        return (_str_at(fact, "period", "type"),)
    if key == "period_value":
        return (_str_at(fact, "period", "value"),)
    if key == "geography_level":
        return (_str_at(fact, "geography", "level"),)
    if key == "geography_id":
        return (_str_at(fact, "geography", "id"),)
    if key == "geography_vintage":
        return (_str_at(fact, "geography", "vintage"),)
    if key == "entity_name":
        return (_str_at(fact, "entity", "name"),)
    if key in {"record_set_id", "layout_record_set_id"}:
        return (_str_at(fact, "layout", "record_set_id"),)
    if key in {"record_set_spec_id", "layout_record_set_spec_id"}:
        return (_str_at(fact, "layout", "record_set_spec_id"),)
    if key in {"groupby_dimension", "layout_groupby_dimension"}:
        return (_str_at(fact, "layout", "groupby_dimension"),)
    if key == "layout_groupby_value_id":
        return (_str_at(fact, "layout", "groupby_value_id"),)
    if key == "layout_measure_id":
        return (_str_at(fact, "layout", "measure_id"),)
    if key == "domain":
        return (_domain(fact),)
    if key == "assertion":
        return (_fact_assertion(fact),)
    raise ValueError(f"Unsupported Ledger fact selector field {key!r}.")


def _measure_id_by_opening_year(expected: object) -> dict[int, str]:
    """Parse a year-to-measure map: four-digit opening years to measure ids."""

    return _by_opening_year(SOURCE_MEASURE_ID_BY_OPENING_YEAR, expected, "measure id")


def _table_by_opening_year(expected: object) -> dict[int, str]:
    """Parse a year-to-table map: four-digit opening years to source tables."""

    return _by_opening_year(SOURCE_TABLE_BY_OPENING_YEAR, expected, "source table")


def _by_opening_year(field: str, expected: object, noun: str) -> dict[int, str]:
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError(
            f"Ledger fact selector field {field!r} must be a non-empty mapping "
            f"of opening year to {noun}."
        )
    by_year: dict[int, str] = {}
    for year, value in expected.items():
        year_text = str(year)
        if (
            isinstance(year, bool)
            or len(year_text) != 4
            or not year_text.isdigit()
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError(
                f"Ledger fact selector field {field!r} maps {year!r} to "
                f"{value!r}; expected a four-digit opening year and a non-blank "
                f"{noun}."
            )
        by_year[int(year_text)] = value
    return by_year


def _measure_by_opening_year_matches(fact: object, expected: object) -> bool:
    """Match an annual fact only to the measure its own opening year declares."""

    return _matches_its_opening_year(
        fact, _measure_id_by_opening_year(expected), "source_measure_id"
    )


def _table_by_opening_year_matches(fact: object, expected: object) -> bool:
    """Match an annual fact only to the table its own opening year declares."""

    return _matches_its_opening_year(
        fact, _table_by_opening_year(expected), "source_table"
    )


def _matches_its_opening_year(
    fact: object, by_year: Mapping[int, str], field: str
) -> bool:
    period_key = _period_key(fact)
    if not period_key[0] or period_key[1] % 100 != 99:
        return False
    declared = by_year.get(period_key[1] // 100)
    return declared is not None and declared in _selector_candidates(fact, field)


def _dimensions_match(fact: object, expected: object) -> bool:
    if isinstance(expected, list):
        expected_names = {str(name) for name in expected}
        dimension_names = {str(key) for key in _dimensions(fact)}
        return dimension_names == expected_names
    if not isinstance(expected, Mapping):
        raise ValueError(
            "Ledger fact selector field 'dimensions' must be a mapping or list."
        )
    dimensions = {str(key): str(value) for key, value in _dimensions(fact).items()}
    return all(
        dimensions.get(str(key)) == str(value) for key, value in expected.items()
    )


def _dimension_values_match(fact: object, expected: object) -> bool:
    """Match dimension-value pins with strict typed equality.

    A pin whose type disagrees with the fact's dimension value (float ``5.0``
    against int ``5``, ``"5"`` against ``5``) does not error — the fact simply
    fails to match, so under a resolves-or-defers activation rule the target
    quietly drops out of the active subset. When a target is unexpectedly
    inactive, check the authoring-run membership report before suspecting the
    feed. Empty pin lists are refused for the same reason as empty membership
    lists in :func:`_fact_matches_selector`.
    """

    if not isinstance(expected, Mapping):
        raise ValueError(
            "Ledger fact selector field 'dimension_values' must be a mapping."
        )
    dimensions = _dimensions(fact)
    for key, expected_value in expected.items():
        if isinstance(expected_value, list) and not expected_value:
            raise ValueError(
                f"dimension_values pin {key!r} is an empty list: an empty pin "
                "list matches nothing while reading like match-anything. Pin "
                "at least one value or drop the dimension."
            )
        if key not in dimensions:
            return False
        actual_value = dimensions[key]
        if isinstance(expected_value, list):
            if not any(
                _strict_scalar_equal(actual_value, candidate)
                for candidate in expected_value
            ):
                return False
            continue
        if not _strict_scalar_equal(actual_value, expected_value):
            return False
    return True


def _strict_scalar_equal(actual: object, expected: object) -> bool:
    actual_type = float if type(actual) is LexicalFloat else type(actual)
    expected_type = float if type(expected) is LexicalFloat else type(expected)
    return actual_type is expected_type and actual == expected


def _measure_concepts(fact: object) -> tuple[str, ...]:
    concepts = (
        _str_at(fact, "measure", "concept"),
        _str_at(fact, "concept_alignment", "canonical_concept"),
        _str_at(fact, "observed_measure", "source_concept"),
        _str_at(fact, "measure", "source_concept"),
    )
    return tuple(dict.fromkeys(concept for concept in concepts if concept))


def _primary_measure_concept(fact: object) -> str:
    concepts = _measure_concepts(fact)
    return concepts[0] if concepts else ""


def _source_measure_concept(fact: object) -> str:
    return (
        _str_at(fact, "observed_measure", "source_concept")
        or _str_at(fact, "measure", "source_concept")
        or _primary_measure_concept(fact)
    )


def _source_name(fact: object) -> str:
    return _str_at(fact, "source", "source_name") or _str_at(
        fact, "observed_measure", "source_name"
    )


def _dimensions(fact: object) -> Mapping[str, object]:
    filters = _at(fact, "filters")
    if isinstance(filters, Mapping):
        return filters
    dimensions = _at(fact, "dimensions")
    if isinstance(dimensions, Mapping):
        return dimensions
    return {}


def _domain(fact: object) -> str:
    return _str_at(fact, "domain") or _str_at(fact, "universe_constraints", "domain")


def _constraint_rows(fact: object) -> tuple[object, ...]:
    constraints = _at(fact, "constraints")
    if isinstance(constraints, list | tuple):
        return tuple(constraints)
    universe_constraints = _at(fact, "universe_constraints", "constraints")
    if isinstance(universe_constraints, list | tuple):
        return tuple(universe_constraints)
    return ()


def _requires_filter_mapping(fact: object) -> bool:
    return _requires_detail_filter_mapping(fact) or _requires_domain_filter_mapping(
        fact
    )


def _requires_detail_filter_mapping(fact: object) -> bool:
    if _constraint_rows(fact):
        return True
    return any(
        not _is_unscoped_dimension_value(value) for value in _dimensions(fact).values()
    )


def _requires_domain_filter_mapping(fact: object) -> bool:
    return not _is_unscoped_domain(_domain(fact))


def _is_unscoped_domain(domain: str) -> bool:
    normalized = domain.strip().lower().replace("_", " ")
    return normalized in {"", "all", "all households", "all persons", "population"}


def _is_unscoped_dimension_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    normalized = str(value).strip().lower().replace("_", " ")
    return normalized in {"", "all", "all returns", "total", "totals", "overall"}


def _allows_signed_target(fact: object, mapping: LedgerTargetMapping) -> bool:
    source_record_id = _source_record_id(fact)
    if source_record_id in mapping.signed_by_source_record_id:
        return True
    return any(
        concept in mapping.signed_by_concept for concept in _measure_concepts(fact)
    )


def _source_citation(fact: object) -> str:
    parts = [
        _source_name(fact),
        _str_at(fact, "source", "source_table")
        or _str_at(fact, "observed_measure", "source_table"),
        _str_at(fact, "source", "source_file"),
        _str_at(fact, "source", "vintage"),
        _str_at(fact, "source", "url"),
    ]
    citation = " | ".join(part for part in parts if part)
    return citation or "PolicyEngine Ledger source-backed fact"


def _ledger_metadata(fact: object, *, fact_key: str) -> dict[str, str]:
    metadata = {
        "ledger_source": "policyengine-ledger-data",
        "ledger_fact_key": fact_key,
        "ledger_source_record_id": _source_record_id(fact),
        "ledger_aggregate_fact_key": _str_at(fact, "aggregate_fact_key"),
        "ledger_semantic_fact_key": _str_at(fact, "semantic_fact_key"),
        "ledger_legacy_fact_key": _str_at(fact, "legacy_fact_key"),
        "ledger_observed_measure_key": _str_at(fact, "observed_measure_key"),
        "ledger_dimension_set_key": _str_at(fact, "dimension_set_key"),
        "ledger_universe_constraint_set_key": _str_at(
            fact, "universe_constraint_set_key"
        ),
        "ledger_measure_concept": _primary_measure_concept(fact),
        "ledger_measure_unit": _str_at(fact, "measure", "unit")
        or _str_at(fact, "observed_measure", "unit"),
        "ledger_source_concept": _source_measure_concept(fact),
        "ledger_concept_relation": _str_at(fact, "measure", "concept_relation")
        or _str_at(fact, "concept_alignment", "relation"),
        "ledger_concept_authority": _str_at(fact, "measure", "concept_authority")
        or _str_at(fact, "concept_alignment", "authority"),
        "ledger_legal_vintage": _str_at(fact, "measure", "legal_vintage")
        or _str_at(fact, "concept_alignment", "legal_vintage"),
        "ledger_period_type": _str_at(fact, "period", "type"),
        "ledger_fact_period": str(_comparable_period_value(fact)),
        # Stamped only when the publisher's label differs from the comparable
        # period (DfT's March-end fiscal labels); absent otherwise.
        "ledger_fact_period_label": (
            _str_at(fact, "period", "value")
            if str(_comparable_period_value(fact)) != _str_at(fact, "period", "value")
            else ""
        ),
        # Recorded only when the fact asserts it; legacy rows that omit the
        # field are not stamped (readers treat absence as
        # observation-by-default, same as the artifact loader).
        "ledger_assertion": _str_at(fact, "assertion"),
        "ledger_geography_level": _str_at(fact, "geography", "level"),
        "ledger_geography_id": _str_at(fact, "geography", "id"),
        "ledger_geography_name": _str_at(fact, "geography", "name"),
        "ledger_geography_vintage": _str_at(fact, "geography", "vintage"),
        "ledger_fact_label": _str_at(fact, "label"),
        "ledger_entity_name": _str_at(fact, "entity", "name"),
        "ledger_entity_role": _str_at(fact, "entity", "role"),
        "ledger_domain": _domain(fact),
        "ledger_layout_record_set_id": _str_at(fact, "layout", "record_set_id"),
        "ledger_layout_record_set_spec_id": _str_at(
            fact, "layout", "record_set_spec_id"
        ),
        "ledger_layout_groupby_dimension": _str_at(fact, "layout", "groupby_dimension"),
        "ledger_layout_groupby_value_id": _str_at(fact, "layout", "groupby_value_id"),
        "ledger_layout_groupby_value_label": _str_at(
            fact, "layout", "groupby_value_label"
        ),
        "ledger_layout_measure_id": _str_at(fact, "layout", "measure_id"),
        "ledger_aggregation_method": _str_at(fact, "aggregation", "method"),
    }
    constraint_rows = _constraint_rows(fact)
    if constraint_rows:
        metadata["ledger_universe_constraint_count"] = str(len(constraint_rows))
    dimensions = _dimensions(fact)
    for key, value in sorted(dimensions.items()):
        if value is not None:
            metadata[f"ledger_filter_{key}"] = str(value)
    for key, value in sorted(_constraint_bound_filters(fact, dimensions).items()):
        metadata.setdefault(f"ledger_filter_{key}", value)
    return {key: value for key, value in metadata.items() if value}


#: Constraint operators that publish a numeric band edge, and the
#: ``ledger_filter_<variable><suffix>`` key each one is stamped under. The
#: ``_lower_bound`` suffix is the one the band materialization reads
#: (``target_materialization._band_lower_edge``); the others record the
#: publisher's upper edge and its openness for readers and receipts.
_CONSTRAINT_BOUND_SUFFIXES: Mapping[str, str] = {
    ">=": "_lower_bound",
    ">": "_lower_bound_exclusive",
    "<": "_upper_bound",
    "<=": "_upper_bound_inclusive",
}


def _constraint_bound_filters(
    fact: object, dimensions: Mapping[str, object]
) -> dict[str, str]:
    """Numeric band edges a fact declares as universe constraints.

    Some publishers state a band as a categorical dimension plus explicit
    numeric constraints (HMRC's CGT size-of-gain and age tables:
    ``cgt_gain_band == gain_3000_to_5999`` with ``cgt_gain >= 3000`` and
    ``cgt_gain < 6000``) rather than as a ``*_lower_bound`` dimension the way
    the SPI income-band tables do. The edges are stamped under the same
    ``ledger_filter_`` vocabulary so a banded measure can slice on them. A
    variable that is already a dimension key is left to the dimension stamp;
    non-numeric or non-filter constraints are ignored.
    """

    bounds: dict[str, str] = {}
    for row in _constraint_rows(fact):
        role = _str_at(row, "role") or "filter"
        if role != "filter":
            continue
        variable = _str_at(row, "variable")
        operator = _str_at(row, "operator")
        suffix = _CONSTRAINT_BOUND_SUFFIXES.get(operator)
        if not variable or suffix is None or variable in dimensions:
            continue
        value = _at(row, "value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(float(value)):
            continue
        bounds[f"{variable}{suffix}"] = str(value)
    return bounds


def constraint_bound_shadowing_dimensions(
    fact: object, variable: str
) -> tuple[str, ...]:
    """Dimensions that can put an operator-less value under *variable*'s bound keys.

    :func:`_constraint_bound_filters` keeps the operator in the key it stamps
    (``>=`` as ``<variable>_lower_bound``, ``<`` as ``<variable>_upper_bound``,
    and so on through ``_CONSTRAINT_BOUND_SUFFIXES``), but
    :func:`_ledger_metadata` stamps the fact's dimensions first and the
    constraint edges only through ``setdefault``. So a dimension named for one
    of those keys is what ``ledger_filter_<that key>`` carries, with no
    operator at all; and a dimension named *variable* makes
    :func:`_constraint_bound_filters` skip every one of its rows, so any bound
    key for it in the metadata is a dimension's. Returns the names of such
    dimensions present on *fact*, by key whatever their value, sorted. Empty
    means every ``ledger_filter_<variable>_*bound*`` key the fact's metadata
    carries was stamped from a constraint row, with the operator its suffix
    names.
    """

    dimensions = _dimensions(fact)
    names = (
        variable,
        *(f"{variable}{suffix}" for suffix in _CONSTRAINT_BOUND_SUFFIXES.values()),
    )
    return tuple(sorted(name for name in names if name in dimensions))


def _diagnostic_target_label_metadata(
    facts: tuple[object, ...],
    *,
    target_period: object,
) -> dict[str, str]:
    """Use a Chronicle fact label only when it still describes the target."""

    if len(facts) != 1:
        return {}
    fact = facts[0]
    label = _str_at(fact, "label").strip()
    fact_period = _at(fact, "period", "value")
    if not label or fact_period is None:
        return {}
    if not period_values_semantically_equal(
        target_period,
        fact_period,
        declared_type=_str_at(fact, "period", "type"),
    ):
        return {}
    return {"diagnostic_target_label": label}


def _hierarchy_rule_enabled(
    rule: LedgerTargetHierarchyRule,
    context: Mapping[str, object],
) -> bool:
    for key, expected in rule.enabled_when.items():
        if _profile_scalar(context.get(key)) != expected:
            return False
    return True


def _hierarchy_parent_key(
    spec: TargetSpec,
    rule: LedgerTargetHierarchyRule,
) -> tuple[str, ...]:
    return _hierarchy_match_key(spec, rule) + (
        spec.metadata.get("ledger_geography_id", ""),
    )


def _hierarchy_match_key(
    spec: TargetSpec,
    rule: LedgerTargetHierarchyRule,
) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in rule.match_spec_fields:
        if not hasattr(spec, field_name):
            raise ValueError(
                f"Ledger target hierarchy rule {rule.rule_id!r} references "
                f"unknown TargetSpec field {field_name!r}."
            )
        values.append(_profile_scalar(getattr(spec, field_name)))
    for key in rule.match_metadata_keys:
        values.append(spec.metadata.get(key, ""))
    return tuple(values)


def _hierarchy_parent_geography_id(
    spec: TargetSpec,
    rule: LedgerTargetHierarchyRule,
) -> str:
    values = {
        **{key: _profile_scalar(value) for key, value in spec.metadata.items()},
        "name": spec.name,
        "entity": spec.entity,
        "period": _profile_scalar(spec.period),
        "family": spec.family,
        "filter": spec.filter or "",
    }
    try:
        return rule.parent_geography_id_template.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} cannot expand "
            f"parent geography template {rule.parent_geography_id_template!r}; "
            f"missing metadata key {exc.args[0]!r} on target {spec.name!r}."
        ) from exc


def _hierarchy_completeness(
    parent: TargetSpec,
    children: list[tuple[int, TargetSpec]],
    rule: LedgerTargetHierarchyRule,
) -> dict[str, object] | None:
    if not rule.expected_child_count_by_parent_key:
        return None
    parent_key = parent.metadata.get(rule.parent_key_metadata_key, "")
    if not parent_key:
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} cannot check child "
            f"completeness for parent {parent.name!r}: missing parent metadata "
            f"{rule.parent_key_metadata_key!r}."
        )
    expected_child_count = rule.expected_child_count_by_parent_key.get(parent_key)
    if expected_child_count is None:
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} has no expected child "
            f"count for parent key {parent_key!r}."
        )
    child_id_values = tuple(
        spec.metadata.get(rule.child_id_metadata_key, "") for _, spec in children
    )
    if "" in child_id_values:
        missing = [
            spec.name
            for _, spec in children
            if not spec.metadata.get(rule.child_id_metadata_key, "")
        ]
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} cannot check child "
            f"completeness for parent {parent.name!r}: missing child metadata "
            f"{rule.child_id_metadata_key!r} on targets {missing!r}."
        )
    child_ids = tuple(sorted(set(child_id_values)))
    if len(child_ids) != len(child_id_values):
        duplicate_child_ids = sorted(
            child_id
            for child_id in set(child_id_values)
            if child_id_values.count(child_id) > 1
        )
        raise ValueError(
            f"Ledger target hierarchy rule {rule.rule_id!r} found duplicate "
            f"child ids for parent {parent.name!r}: {duplicate_child_ids!r}."
        )
    observed_child_count = len(child_ids)
    return {
        "parent_key": parent_key,
        "expected_child_count": expected_child_count,
        "observed_child_count": observed_child_count,
        "child_ids": ",".join(child_ids),
        "complete": observed_child_count == expected_child_count,
    }


def _hierarchy_completeness_metadata(
    completeness: Mapping[str, object] | None,
) -> dict[str, str]:
    if completeness is None:
        return {}
    return {
        "hierarchy_parent_key": _profile_scalar(completeness["parent_key"]),
        "hierarchy_expected_child_count": _profile_scalar(
            completeness["expected_child_count"]
        ),
        "hierarchy_observed_child_count": _profile_scalar(
            completeness["observed_child_count"]
        ),
        "hierarchy_child_ids": _profile_scalar(completeness["child_ids"]),
    }


def _profile_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _format_float(value: float) -> str:
    return f"{value:.15g}"


def _registry_digest(
    registry: TargetRegistry,
    *,
    payload_fn: Any = asdict,
) -> str:
    payload = {
        "country": registry.country,
        "specs": [payload_fn(spec) for spec in registry.specs],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _calibration_effective_spec_payload(spec: TargetSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "entity": spec.entity,
        "measure": spec.measure,
        "value": spec.value,
        "filter": spec.filter,
        "period": spec.period,
        "tolerance": spec.tolerance,
    }


def _first_spec_differences(
    expected: TargetRegistry,
    actual: TargetRegistry,
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    expected_by_key = {spec.key: asdict(spec) for spec in expected.specs}
    actual_by_key = {spec.key: asdict(spec) for spec in actual.specs}
    failures: list[str] = []
    missing = sorted(set(expected_by_key) - set(actual_by_key))
    extra = sorted(set(actual_by_key) - set(expected_by_key))
    if missing:
        failures.append(f"missing target keys: {missing[:limit]!r}.")
    if extra:
        failures.append(f"extra target keys: {extra[:limit]!r}.")
    for key in sorted(set(expected_by_key) & set(actual_by_key)):
        if expected_by_key[key] != actual_by_key[key]:
            failures.append(f"target {key!r} differs.")
        if len(failures) >= limit:
            break
    return tuple(failures)
