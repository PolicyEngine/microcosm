"""Pre-calibration agreement contract for assembled household spines.

The assembly seam records each entity row's source spine in
``{entity}_support_channel`` and a deeply frozen frame receipt. This gate
validates the live columns against that receipt, then intentionally conditions
on the provenance: it compares the weighted distribution produced by the
common operator pass across every pair of source spines, before calibration
can conceal a disagreement.

For every registered transferred, imputed, seeded, or simulated surface,
agreement has two parts:

* the ratio of weighted nonzero incidence rates must lie in ``[0.8, 1.25]``;
* at weighted conditional quantiles 10, 25, 50, 75, and 90 percent, the
  maximum symmetric relative distance must be no greater than ``0.25``.

Categorical surfaces instead compare the complete weighted category
distribution by total-variation distance, which must be no greater than
``0.25``. Declared joint categorical groups are additionally compared as
tuples, so equal marginals cannot conceal invalid cross-column combinations.

The symmetric quantile distance at one quantile is
``2 * abs(left - right) / (abs(left) + abs(right))``.  It is zero when both
quantiles are zero and at most two otherwise.  Quantiles use positive-weight,
nonzero observations and the inverse empirical CDF.  These declarations are
the initial #395 gate specification; they are constants, not fitted to a
particular build output.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_transfer import (
    TargetFamilies,
    acs_derived_transfer_expectations,
    declared_acs_transfer_target_families,
)
from microcosm.build.us_runtime.support_provenance import (
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_source_id_column,
    validate_assembly_provenance,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract

if TYPE_CHECKING:
    US_SPINE_AGREEMENT_REGISTRY: tuple[SpineAgreementSpec, ...]

__all__ = [
    "DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE",
    "DEFAULT_INCIDENCE_RATIO_BOUNDS",
    "DEFAULT_QUANTILE_ENVELOPE_TOLERANCE",
    "DEFAULT_SPINE_AGREEMENT_QUANTILES",
    "SpineAgreementSpec",
    "US_SPINE_AGREEMENT_REGISTRY",
    "default_spine_agreement_registry",
    "normalize_transfer_family_name",
    "spine_agreement_gate",
    "validate_spine_agreement_registry",
]

DEFAULT_INCIDENCE_RATIO_BOUNDS = (0.8, 1.25)
DEFAULT_SPINE_AGREEMENT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_QUANTILE_ENVELOPE_TOLERANCE = 0.25
DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE = 0.25
_BATCH_SEPARATOR = "__batch_"
_DERIVED_FAMILY = "derived_transfer"
_SIMULATED_OUTPUT_FAMILY = "simulated_output"
_TAKE_UP_FAMILY = "take_up"
_GATE_NAME = "us_spine_agreement"
_SIMULATED_OUTPUTS: Mapping[str, tuple[str, ...]] = {
    "person": ("ssi",),
}
_JOINT_CATEGORICAL_TARGET_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ssn_card_type", "immigration_status_str"),
)


class _ResolvedWeights(Protocol):
    values: np.ndarray


class _AgreementFrame(Protocol):
    entities: Sequence[str]

    def table(self, entity: str) -> pd.DataFrame: ...

    def resolve_weights(self, entity: str) -> _ResolvedWeights: ...


@dataclass(frozen=True)
class SpineAgreementSpec:
    """Distribution checks for one entity-scoped transfer family.

    ``family`` is the canonical family name: any QRF ``__batch_N`` suffix is
    removed before a specification enters the registry.
    """

    entity: str
    family: str
    columns: tuple[str, ...]
    joint_categorical_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entity, str) or not self.entity.strip():
            raise ValueError("SpineAgreementSpec.entity must be a non-empty string.")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("SpineAgreementSpec.family must be a non-empty string.")
        if _BATCH_SEPARATOR in self.family:
            raise ValueError(
                "SpineAgreementSpec.family must be canonical; normalize "
                f"{self.family!r} before registering it."
            )
        if not isinstance(self.columns, tuple):
            raise ValueError("SpineAgreementSpec.columns must be an immutable tuple.")
        if not self.columns:
            raise ValueError(
                f"SpineAgreementSpec {self.entity}/{self.family} has no columns."
            )
        if any(
            not isinstance(column, str) or not column.strip() for column in self.columns
        ):
            raise ValueError(
                f"SpineAgreementSpec {self.entity}/{self.family} has an invalid "
                "column name."
            )
        duplicate_columns = sorted(
            {column for column in self.columns if self.columns.count(column) > 1}
        )
        if duplicate_columns:
            raise ValueError(
                f"SpineAgreementSpec {self.entity}/{self.family} repeats columns "
                f"{duplicate_columns}."
            )
        if not isinstance(self.joint_categorical_groups, tuple) or any(
            not isinstance(group, tuple)
            for group in self.joint_categorical_groups
        ):
            raise ValueError(
                "SpineAgreementSpec.joint_categorical_groups must be an "
                "immutable tuple of tuples."
            )
        invalid_groups = [
            group
            for group in self.joint_categorical_groups
            if len(group) < 2
            or any(
                not isinstance(column, str) or not column.strip()
                for column in group
            )
            or len(set(group)) != len(group)
            or not set(group).issubset(self.columns)
        ]
        if invalid_groups:
            raise ValueError(
                f"SpineAgreementSpec {self.entity}/{self.family} has invalid "
                f"joint categorical group(s): {invalid_groups}."
            )
        grouped_columns = [
            column
            for group in self.joint_categorical_groups
            for column in group
        ]
        repeated_group_columns = sorted(
            {
                column
                for column in grouped_columns
                if grouped_columns.count(column) > 1
            }
        )
        if repeated_group_columns:
            raise ValueError(
                f"SpineAgreementSpec {self.entity}/{self.family} repeats columns "
                "across joint categorical groups: "
                f"{repeated_group_columns}."
            )


def normalize_transfer_family_name(family: str) -> str:
    """Remove a QRF batching suffix from one transfer family name."""

    if not isinstance(family, str) or not family.strip():
        raise ValueError("Transfer family names must be non-empty strings.")
    if _BATCH_SEPARATOR not in family:
        return family
    canonical, batch = family.rsplit(_BATCH_SEPARATOR, 1)
    if not canonical or not batch.isdigit() or int(batch) < 1:
        raise ValueError(
            f"Malformed transfer-family batch suffix in {family!r}; expected "
            "'<family>__batch_<positive integer>'."
        )
    return canonical


def default_spine_agreement_registry(
    target_families: TargetFamilies | None = None,
) -> tuple[SpineAgreementSpec, ...]:
    """Build the complete checked-distribution registry.

    Split QRF families are merged back to their canonical family.  Columns
    derived deterministically from transferred parents are registered under
    ``derived_transfer``.  The checked-in take-up inventory supplies every
    take-up input, including stages owned outside the generic seeder, and
    simulation outputs measured by the multispine QA contract are registered
    separately.  Thus the pre-calibration gate covers the full chartered
    surface, not only fitted transfer leaves.
    """

    families = (
        declared_acs_transfer_target_families()
        if target_families is None
        else target_families
    )
    normalized = _declared_agreement_surface(families)

    registry = tuple(
        SpineAgreementSpec(
            entity=entity,
            family=family,
            columns=tuple(dict.fromkeys(columns)),
            joint_categorical_groups=tuple(
                group
                for group in _JOINT_CATEGORICAL_TARGET_GROUPS
                if set(group).issubset(columns)
            ),
        )
        for (entity, family), columns in sorted(normalized.items())
    )
    validate_spine_agreement_registry(registry, target_families=families)
    return registry


def validate_spine_agreement_registry(
    registry: Sequence[SpineAgreementSpec],
    *,
    target_families: TargetFamilies | None = None,
) -> tuple[SpineAgreementSpec, ...]:
    """Validate a registry, optionally requiring exact charter coverage.

    Structural mistakes raise :class:`ValueError`; observed distribution
    disagreements belong to :func:`spine_agreement_gate` and become batched
    :class:`~microcosm.build.gates.GateResult` failures instead.
    """

    if isinstance(registry, (str, bytes)) or not isinstance(registry, Sequence):
        raise ValueError("Spine-agreement registry must be a sequence of specs.")
    normalized_registry = tuple(registry)
    if not normalized_registry:
        raise ValueError("Spine-agreement registry must not be empty.")
    if any(not isinstance(spec, SpineAgreementSpec) for spec in normalized_registry):
        raise ValueError(
            "Spine-agreement registry entries must be SpineAgreementSpec values."
        )
    keys = [(spec.entity, spec.family) for spec in normalized_registry]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(
            f"Spine-agreement registry repeats entity/family entries: {duplicates}."
        )

    column_owners: dict[tuple[str, str], str] = {}
    for spec in normalized_registry:
        for column in spec.columns:
            key = (spec.entity, column)
            previous = column_owners.get(key)
            if previous is not None:
                raise ValueError(
                    f"Spine-agreement column {spec.entity}/{column} is registered "
                    f"by both {previous!r} and {spec.family!r}."
                )
            column_owners[key] = spec.family

    if target_families is not None:
        expected = _declared_agreement_surface(target_families)
        expected_surface = {
            key: frozenset(columns) for key, columns in expected.items()
        }
        observed_surface = {
            (spec.entity, spec.family): frozenset(spec.columns)
            for spec in normalized_registry
        }
        if observed_surface != expected_surface:
            missing = sorted(
                (entity, family, column)
                for (entity, family), columns in expected_surface.items()
                for column in columns
                - observed_surface.get((entity, family), frozenset())
            )
            extra = sorted(
                (entity, family, column)
                for (entity, family), columns in observed_surface.items()
                for column in columns
                - expected_surface.get((entity, family), frozenset())
            )
            missing_families = sorted(set(expected_surface) - set(observed_surface))
            extra_families = sorted(set(observed_surface) - set(expected_surface))
            raise ValueError(
                "Spine-agreement registry does not exactly cover the chartered "
                f"surface; missing_columns={missing}, extra_columns={extra}, "
                f"missing_families={missing_families}, "
                f"extra_families={extra_families}."
            )
    return normalized_registry


def spine_agreement_gate(
    frame: _AgreementFrame,
    *,
    registry: Sequence[SpineAgreementSpec] | None = None,
) -> GateResult:
    """Evaluate every registered distribution across every source-spine pair.

    Registry defects raise :class:`ValueError`.  Missing or invalid frame
    evidence and every out-of-tolerance statistic are accumulated into one
    failed result, following the gate batching convention.
    """

    specs = (
        _canonical_spine_agreement_registry()
        if registry is None
        else validate_spine_agreement_registry(registry)
    )
    frame_entities = tuple(frame.entities)
    validate_assembly_provenance(
        frame,
        boundary="spine agreement gate",
        require_manifest=any(
            spine_source_id_column(entity) in frame.table(entity)
            for entity in frame_entities
        ),
    )
    failures: list[str] = []
    comparisons: dict[str, object] = {}
    untestable_comparisons: list[str] = []
    contexts: dict[str, tuple[pd.DataFrame, np.ndarray, tuple[str, ...]] | None] = {}

    for entity in sorted({spec.entity for spec in specs}):
        if entity not in frame_entities:
            failures.append(f"{entity}: registered entity is absent from the frame.")
            contexts[entity] = None
            continue
        table = frame.table(entity)
        provenance_column = f"{entity}_support_channel"
        if provenance_column not in table:
            failures.append(
                f"{entity}: source-spine provenance column "
                f"{provenance_column!r} is absent."
            )
            contexts[entity] = None
            continue
        provenance = table[provenance_column]
        invalid_provenance = provenance.isna() | provenance.astype(str).str.strip().eq(
            ""
        )
        if invalid_provenance.any():
            failures.append(
                f"{entity}: {provenance_column} has "
                f"{int(invalid_provenance.sum())} missing/empty value(s)."
            )
            contexts[entity] = None
            continue
        channels = tuple(sorted(provenance.astype(str).unique()))
        if PUF_TAX_DETAIL_SUPPORT_CHANNEL in channels:
            failures.append(
                f"{entity}: {PUF_TAX_DETAIL_SUPPORT_CHANNEL!r} is a clone role, "
                "not valid receipt-declared source-spine provenance."
            )
            contexts[entity] = None
            continue
        if len(channels) < 2:
            failures.append(
                f"{entity}: agreement requires at least two source spines; "
                f"observed {list(channels)}."
            )
            contexts[entity] = None
            continue

        try:
            weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            failures.append(
                f"{entity}: resolved weights are not a numeric vector: {error}."
            )
            contexts[entity] = None
            continue
        if len(weights) != len(table):
            failures.append(
                f"{entity}: resolved {len(weights)} weights for {len(table)} rows."
            )
            contexts[entity] = None
            continue
        invalid_weights = ~np.isfinite(weights) | (weights < 0.0)
        if invalid_weights.any():
            failures.append(
                f"{entity}: resolved weights have "
                f"{int(invalid_weights.sum())} non-finite/negative value(s)."
            )
            contexts[entity] = None
            continue
        channel_values = provenance.astype(str).to_numpy()
        zero_mass_channels = [
            channel
            for channel in channels
            if float(weights[channel_values == channel].sum()) <= 0.0
        ]
        if zero_mass_channels:
            failures.append(
                f"{entity}: source spines have zero resolved weight: "
                f"{zero_mass_channels}."
            )
            contexts[entity] = None
            continue
        contexts[entity] = (table, weights, channels)

    observed_source_channels = {
        entity: list(context[2])
        for entity, context in sorted(contexts.items())
        if context is not None
    }
    expected_source_channels = tuple(
        sorted(
            {
                channel
                for channels in observed_source_channels.values()
                for channel in channels
            }
        )
    )
    missing_source_channels_by_entity: dict[str, list[str]] = {}
    for entity, observed_channels in observed_source_channels.items():
        missing_channels = sorted(
            set(expected_source_channels) - set(observed_channels)
        )
        if not missing_channels:
            continue
        missing_source_channels_by_entity[entity] = missing_channels
        failures.append(
            f"{entity}: source-spine set is inconsistent across registered "
            f"entity grains; observed {observed_channels}, expected "
            f"{list(expected_source_channels)}, missing {missing_channels}."
        )

    checked_columns = 0
    checked_pairs = 0
    tested_pairs = 0
    checked_joint_categorical_groups = 0
    for spec in specs:
        context = contexts.get(spec.entity)
        if context is None:
            continue
        table, weights, observed_channels = context
        channel_values = table[f"{spec.entity}_support_channel"].astype(str).to_numpy()
        comparison_channels = (
            expected_source_channels
            if len(expected_source_channels) >= 2
            else observed_channels
        )
        for column in spec.columns:
            label = f"{spec.entity}/{spec.family}/{column}"
            if column not in table:
                failures.append(f"{label}: registered column is absent from the frame.")
                continue
            series = table[column]
            relevant = weights > 0.0
            if not (is_numeric_dtype(series) or is_bool_dtype(series)):
                invalid_values = relevant & series.isna().to_numpy(dtype=bool)
                if invalid_values.any():
                    failures.append(
                        f"{label}: {int(invalid_values.sum())} positive-weight "
                        "categorical value(s) are missing."
                    )
                    continue
                category_values = series.to_numpy(dtype=object)
                unhashable = sum(
                    1
                    for value, included in zip(
                        category_values,
                        relevant,
                        strict=True,
                    )
                    if included and not _is_hashable(value)
                )
                if unhashable:
                    failures.append(
                        f"{label}: {unhashable} positive-weight categorical "
                        "value(s) are not hashable."
                    )
                    continue
                checked_columns += 1
                pair_count, tested_count = _record_categorical_comparisons(
                    label=label,
                    values=category_values,
                    weights=weights,
                    channel_values=channel_values,
                    observed_channels=observed_channels,
                    comparison_channels=comparison_channels,
                    failures=failures,
                    comparisons=comparisons,
                    untestable_comparisons=untestable_comparisons,
                )
                checked_pairs += pair_count
                tested_pairs += tested_count
                continue

            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            invalid_values = relevant & ~np.isfinite(values)
            if invalid_values.any():
                failures.append(
                    f"{label}: {int(invalid_values.sum())} positive-weight value(s) "
                    "are missing or non-finite."
                )
                continue

            checked_columns += 1
            lower, upper = DEFAULT_INCIDENCE_RATIO_BOUNDS
            for left_channel, right_channel in itertools.combinations(
                comparison_channels, 2
            ):
                checked_pairs += 1
                comparison_key = f"{label}/{left_channel}_vs_{right_channel}"
                missing_pair_channels = sorted(
                    {left_channel, right_channel} - set(observed_channels)
                )
                if missing_pair_channels:
                    comparisons[comparison_key] = {
                        "status": "untestable_missing_source_spine",
                        "missing_source_spines": missing_pair_channels,
                    }
                    untestable_comparisons.append(comparison_key)
                    continue
                left = channel_values == left_channel
                right = channel_values == right_channel
                left_incidence = _weighted_incidence(values[left], weights[left])
                right_incidence = _weighted_incidence(values[right], weights[right])
                if left_incidence == 0.0 and right_incidence == 0.0:
                    comparisons[comparison_key] = {
                        "status": "untestable_both_zero",
                        "left_incidence": left_incidence,
                        "right_incidence": right_incidence,
                        "incidence_ratio_right_over_left": None,
                        "quantile_envelope_distance": None,
                    }
                    untestable_comparisons.append(comparison_key)
                    failures.append(
                        f"{comparison_key}: both source spines have zero weighted "
                        "nonzero incidence; the registered comparison is untestable."
                    )
                    continue

                tested_pairs += 1
                incidence_ratio = _incidence_ratio(left_incidence, right_incidence)
                left_quantiles = _conditional_nonzero_quantiles(
                    values[left], weights[left], DEFAULT_SPINE_AGREEMENT_QUANTILES
                )
                right_quantiles = _conditional_nonzero_quantiles(
                    values[right], weights[right], DEFAULT_SPINE_AGREEMENT_QUANTILES
                )
                envelope_distance = _quantile_envelope_distance(
                    left_quantiles, right_quantiles
                )
                comparisons[comparison_key] = {
                    "status": "tested",
                    "left_incidence": left_incidence,
                    "right_incidence": right_incidence,
                    "incidence_ratio_right_over_left": _manifest_number(
                        incidence_ratio
                    ),
                    "quantile_envelope_distance": _manifest_number(envelope_distance),
                }
                if not lower <= incidence_ratio <= upper:
                    failures.append(
                        f"{comparison_key}: weighted nonzero-incidence ratio "
                        f"{incidence_ratio:.6g} is outside [{lower:.6g}, "
                        f"{upper:.6g}] (left={left_incidence:.6g}, "
                        f"right={right_incidence:.6g})."
                    )
                if envelope_distance > DEFAULT_QUANTILE_ENVELOPE_TOLERANCE:
                    failures.append(
                        f"{comparison_key}: weighted conditional-quantile "
                        f"envelope distance {envelope_distance:.6g} exceeds "
                        f"{DEFAULT_QUANTILE_ENVELOPE_TOLERANCE:.6g}."
                    )

        for group in spec.joint_categorical_groups:
            if any(column not in table for column in group):
                continue
            group_series = [table[column] for column in group]
            if any(
                is_numeric_dtype(series) or is_bool_dtype(series)
                for series in group_series
            ):
                failures.append(
                    f"{spec.entity}/{spec.family}/joint[{','.join(group)}]: "
                    "registered joint categorical columns must all be categorical."
                )
                continue
            relevant = weights > 0.0
            invalid_values = relevant & np.logical_or.reduce(
                [series.isna().to_numpy(dtype=bool) for series in group_series]
            )
            if invalid_values.any():
                continue
            category_values = np.empty(len(table), dtype=object)
            category_values[:] = list(
                zip(
                    *[
                        series.to_numpy(dtype=object)
                        for series in group_series
                    ],
                    strict=True,
                )
            )
            if any(
                included and not _is_hashable(value)
                for value, included in zip(
                    category_values,
                    relevant,
                    strict=True,
                )
            ):
                continue
            checked_joint_categorical_groups += 1
            pair_count, tested_count = _record_categorical_comparisons(
                label=(
                    f"{spec.entity}/{spec.family}/"
                    f"joint[{','.join(group)}]"
                ),
                values=category_values,
                weights=weights,
                channel_values=channel_values,
                observed_channels=observed_channels,
                comparison_channels=comparison_channels,
                failures=failures,
                comparisons=comparisons,
                untestable_comparisons=untestable_comparisons,
            )
            checked_pairs += pair_count
            tested_pairs += tested_count

    return GateResult(
        name=_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            "statistic": {
                "incidence": "weighted share with value != 0",
                "incidence_ratio": (
                    "right incidence / left incidence; undefined when both are zero"
                ),
                "conditional_quantiles": list(DEFAULT_SPINE_AGREEMENT_QUANTILES),
                "quantile_distance": (
                    "max_q 2*abs(left_q-right_q)/(abs(left_q)+abs(right_q))"
                ),
                "categorical_distribution": (
                    "resolved-weight category shares, including registered "
                    "joint tuples"
                ),
                "categorical_distance": (
                    "0.5 * sum_category abs(left_share-right_share)"
                ),
            },
            "tolerances": {
                "incidence_ratio_bounds": list(DEFAULT_INCIDENCE_RATIO_BOUNDS),
                "max_quantile_envelope_distance": (DEFAULT_QUANTILE_ENVELOPE_TOLERANCE),
                "max_categorical_total_variation_distance": (
                    DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE
                ),
            },
            "registered_families": len(specs),
            "checked_columns": checked_columns,
            "checked_joint_categorical_groups": (
                checked_joint_categorical_groups
            ),
            "checked_spine_pairs": checked_pairs,
            "tested_spine_pairs": tested_pairs,
            "untestable_comparisons": sorted(untestable_comparisons),
            "expected_source_channels": list(expected_source_channels),
            "observed_source_channels_by_entity": observed_source_channels,
            "missing_source_channels_by_entity": missing_source_channels_by_entity,
            "comparisons": comparisons,
        },
    )


def _declared_agreement_surface(
    target_families: TargetFamilies,
) -> dict[tuple[str, str], list[str]]:
    """Return transfer, take-up, and simulated-output registry families."""

    normalized = _normalize_target_families(target_families)
    derived = acs_derived_transfer_expectations(target_families)
    for column, entity in sorted(derived.items(), key=lambda item: (item[1], item[0])):
        normalized.setdefault((entity, _DERIVED_FAMILY), []).append(column)

    registered_columns = {
        (entity, column)
        for (entity, _family), columns in normalized.items()
        for column in columns
    }
    from microcosm.build.spec_engine.engine_abi import (
        active_take_up_manifest_program_bindings,
    )

    active_bindings = active_take_up_manifest_program_bindings()
    if active_bindings is None:
        take_up_bindings = tuple(
            (program.variable, program.entity, program.populace_treatment)
            for program in load_take_up_contract().programs
        )
    else:
        take_up_bindings = active_bindings
    for variable, entity, _populace_treatment in sorted(
        take_up_bindings,
        key=lambda item: (item[1], item[0]),
    ):
        key = (entity, variable)
        if key in registered_columns:
            continue
        normalized.setdefault((entity, _TAKE_UP_FAMILY), []).append(variable)
        registered_columns.add(key)

    for entity, columns in sorted(_SIMULATED_OUTPUTS.items()):
        for column in columns:
            key = (entity, column)
            if key in registered_columns:
                continue
            normalized.setdefault((entity, _SIMULATED_OUTPUT_FAMILY), []).append(column)
            registered_columns.add(key)
    return normalized


def _normalize_target_families(
    target_families: TargetFamilies,
) -> dict[tuple[str, str], list[str]]:
    if not isinstance(target_families, Mapping):
        raise ValueError("target_families must map entities to family mappings.")
    normalized: dict[tuple[str, str], list[str]] = {}
    column_owner: dict[tuple[str, str], str] = {}
    for entity, families in target_families.items():
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("Target-family entity names must be non-empty strings.")
        if not isinstance(families, Mapping):
            raise ValueError(f"target_families[{entity!r}] must be a family mapping.")
        for family, raw_columns in families.items():
            canonical_family = normalize_transfer_family_name(family)
            if isinstance(raw_columns, (str, bytes)) or not isinstance(
                raw_columns, Sequence
            ):
                raise ValueError(
                    f"Transfer family {entity}/{family} must contain a sequence "
                    "of column names."
                )
            columns = tuple(raw_columns)
            if not columns:
                raise ValueError(f"Transfer family {entity}/{family} has no columns.")
            for column in columns:
                if not isinstance(column, str) or not column.strip():
                    raise ValueError(
                        f"Transfer family {entity}/{family} has an invalid column."
                    )
                owner_key = (entity, column)
                previous = column_owner.get(owner_key)
                if previous is not None and previous != canonical_family:
                    raise ValueError(
                        f"Transfer column {entity}/{column} belongs to both "
                        f"{previous!r} and {canonical_family!r}."
                    )
                column_owner[owner_key] = canonical_family
            normalized.setdefault((entity, canonical_family), []).extend(columns)
    return normalized


def _record_categorical_comparisons(
    *,
    label: str,
    values: np.ndarray,
    weights: np.ndarray,
    channel_values: np.ndarray,
    observed_channels: tuple[str, ...],
    comparison_channels: tuple[str, ...],
    failures: list[str],
    comparisons: dict[str, object],
    untestable_comparisons: list[str],
) -> tuple[int, int]:
    """Record fixed weighted total-variation checks for one category surface."""

    checked_pairs = 0
    tested_pairs = 0
    for left_channel, right_channel in itertools.combinations(
        comparison_channels,
        2,
    ):
        checked_pairs += 1
        comparison_key = f"{label}/{left_channel}_vs_{right_channel}"
        missing_pair_channels = sorted(
            {left_channel, right_channel} - set(observed_channels)
        )
        if missing_pair_channels:
            comparisons[comparison_key] = {
                "status": "untestable_missing_source_spine",
                "missing_source_spines": missing_pair_channels,
            }
            untestable_comparisons.append(comparison_key)
            continue

        left = channel_values == left_channel
        right = channel_values == right_channel
        left_distribution = _weighted_category_distribution(
            values[left],
            weights[left],
        )
        right_distribution = _weighted_category_distribution(
            values[right],
            weights[right],
        )
        distance = _categorical_total_variation_distance(
            left_distribution,
            right_distribution,
        )
        tested_pairs += 1
        comparisons[comparison_key] = {
            "status": "tested",
            "left_category_shares": _manifest_category_shares(
                left_distribution
            ),
            "right_category_shares": _manifest_category_shares(
                right_distribution
            ),
            "categorical_total_variation_distance": distance,
        }
        if distance > DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE:
            failures.append(
                f"{comparison_key}: weighted categorical total-variation "
                f"distance {distance:.6g} exceeds "
                f"{DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE:.6g}."
            )
    return checked_pairs, tested_pairs


def _weighted_category_distribution(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[object, float]:
    total_weight = float(weights.sum())
    distribution: dict[object, float] = {}
    for value, weight in zip(values, weights, strict=True):
        if weight <= 0.0:
            continue
        distribution[value] = distribution.get(value, 0.0) + float(weight)
    return {
        category: category_weight / total_weight
        for category, category_weight in distribution.items()
    }


def _categorical_total_variation_distance(
    left: Mapping[object, float],
    right: Mapping[object, float],
) -> float:
    categories = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(category, 0.0) - right.get(category, 0.0))
        for category in categories
    )


def _manifest_category_shares(
    distribution: Mapping[object, float],
) -> dict[str, float]:
    return {
        _manifest_category(category): distribution[category]
        for category in sorted(distribution, key=_category_sort_key)
    }


def _manifest_category(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _category_sort_key(value: object) -> tuple[str, str, str]:
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


def _is_hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _weighted_incidence(values: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[values != 0.0].sum() / weights.sum())


def _incidence_ratio(left: float, right: float) -> float:
    if left == 0.0:
        return 1.0 if right == 0.0 else math.inf
    return right / left


def _manifest_number(value: float) -> float | str:
    if math.isinf(value):
        return "infinity" if value > 0.0 else "-infinity"
    return value


def _conditional_nonzero_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...],
) -> np.ndarray | None:
    retained = (values != 0.0) & (weights > 0.0)
    if not retained.any():
        return None
    retained_values = values[retained]
    retained_weights = weights[retained]
    order = np.argsort(retained_values, kind="stable")
    sorted_values = retained_values[order]
    cumulative = np.cumsum(retained_weights[order])
    cumulative /= cumulative[-1]
    positions = np.minimum(
        np.searchsorted(cumulative, np.asarray(quantiles), side="left"),
        len(sorted_values) - 1,
    )
    return sorted_values[positions]


def _quantile_envelope_distance(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> float:
    if left is None or right is None:
        return 0.0 if left is None and right is None else math.inf
    denominator = np.abs(left) + np.abs(right)
    distances = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(np.max(distances))


def _canonical_spine_agreement_registry() -> tuple[SpineAgreementSpec, ...]:
    """Return and cache the typed registry without an eager module cycle."""

    cached = globals().get("US_SPINE_AGREEMENT_REGISTRY")
    if cached is None:
        cached = default_spine_agreement_registry()
        globals()["US_SPINE_AGREEMENT_REGISTRY"] = cached
    return cached


def __getattr__(name: str) -> object:
    """Build the canonical registry only when a consumer requests it.

    ``multispine_pool`` imports the agreement functions while the typed engine
    ABI may, in turn, inspect that pool module.  Deferring the module constant
    keeps that import graph acyclic without weakening the typed contract used
    to construct the registry.
    """

    if name != "US_SPINE_AGREEMENT_REGISTRY":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _canonical_spine_agreement_registry()


def __dir__() -> list[str]:
    """Expose the unresolved registry through normal module discovery."""

    return sorted({*globals(), "US_SPINE_AGREEMENT_REGISTRY"})
