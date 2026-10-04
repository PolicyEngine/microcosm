"""Calibration target-loss weights for US releases, in one implementation.

A US calibration minimizes the weighted mean of each target's capped, scaled
miss. This module computes those weights. The national fiscal release
(``tools/build_us_fiscal_refresh_release.py``) and the ACS local-area release
(``tools/build_us_acs_local_release.py``) both call :func:`target_loss_weights`.
They differ only in the *row mapping*, which tells the formula what each
target is.

The formula (:data:`US_FISCAL_TARGET_LOSS_WEIGHTING`):

1. **Value basis.** Each target is a ``"count"`` (people, households,
   returns, enrollees, recipients) or an ``"amount"`` (dollars).
2. **Value weight.** ``max(|value|, 1) ** 0.5``, divided by that quantity's
   mean over the target's basis.
3. **Concept budget.** Targets that share a concept-budget key form a group,
   whose weights are rescaled to sum to the largest member's weight, keeping
   their proportions. Both mappings make the congressional-district targets
   of one concept in one state a group, so adding district geography cannot
   multiply a concept's weight. Every other target is its own group.
4. **Basis budget.** Each basis present is rescaled to carry an equal share
   of the total weight (half each when both are present).
5. Normalize to mean 1.
6. **Family multipliers** (optional, ``FAMILY=MULTIPLIER``). Each named
   family's weights are multiplied, in sorted family order, and the result is
   renormalized to mean 1. A multiplier that names no target's family is an
   error.

Row mappings:

- :data:`US_FISCAL_TARGET_LOSS_ROW_MAPPING` reads the ledger compiler's
  target metadata: ``measure_mode`` and ``source_measure_id`` for the basis,
  and ``ledger_geography_level``, ``state_fips``, the filter and the other
  semantic metadata for the concept key. It is the national release's
  mapping, moved here unchanged.
- :data:`US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING` classifies every row of the
  ACS local-area surface explicitly and refuses any row it cannot classify.
  Ledger-compiled rows (SNAP, Medicaid, SOI) take the national mapping, and
  their basis must agree with their ``ledger_measure_unit``. The Census ladder
  population rows (``pop_state_*``, ``pop_cd_*``), which carry no ledger
  metadata, are counts, as the national release's Census population targets
  are. Each state's ``pop_cd_*`` rows share one concept budget. District SOI
  rows take the national district key without the metadata that names the
  district (:data:`ACS_LOCAL_PER_DISTRICT_METADATA`), so the districts of one
  concept in one state share one budget; each ``state_cd`` reconciliation
  block must be exactly one group.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The loss formula's identifier, recorded in release diagnostics.
US_FISCAL_TARGET_LOSS_WEIGHTING = (
    "sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct"
)
US_FISCAL_TARGET_VALUE_WEIGHT_POWER = 0.5
#: Metadata that identifies a target's place, record or reconciliation rather
#: than its concept. The concept-budget key of a congressional-district target
#: ignores it, so the districts of one concept in one state share one key.
US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS = frozenset(
    {
        "congressional_district_geoid",
        "geography_scope",
        "hierarchy_child_ids",
        "hierarchy_child_sum_raw",
        "hierarchy_coverage_ratio",
        "hierarchy_expected_child_count",
        "hierarchy_observed_child_count",
        "hierarchy_parent_geography_id",
        "hierarchy_parent_geography_level",
        "hierarchy_parent_key",
        "hierarchy_parent_target_name",
        "hierarchy_parent_target_period",
        "hierarchy_parent_value",
        "hierarchy_raw_value",
        "hierarchy_reconciliation_factor",
        "hierarchy_reconciliation_method",
        "hierarchy_reconciliation_rule",
        "ledger_aggregate_fact_key",
        "ledger_dimension_set_key",
        "ledger_fact_key",
        "ledger_geography_id",
        "ledger_geography_level",
        "ledger_geography_name",
        "ledger_geography_vintage",
        "ledger_legacy_fact_key",
        "ledger_layout_groupby_dimension",
        "ledger_layout_groupby_value_id",
        "ledger_layout_record_set_id",
        "ledger_observed_measure_key",
        "ledger_semantic_fact_key",
        "ledger_source_record_id",
        "state_fips",
    }
)

COUNT_BASIS = "count"
AMOUNT_BASIS = "amount"
CONGRESSIONAL_DISTRICT = "congressional_district"
STATE = "state"


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def target_loss_weights_from_rows(
    values: Sequence[float],
    bases: Sequence[str],
    concept_budget_keys: Sequence[Hashable],
    families: Sequence[str | None],
    family_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Steps 2-6 of the formula, over row-aligned per-target inputs.

    Returns float64 weights, row-aligned with the inputs, positive and of
    mean 1 whenever there is at least one row.
    """

    n_rows = len(values)
    if not len(bases) == len(concept_budget_keys) == len(families) == n_rows:
        raise ValueError(
            "Target-loss inputs are not row-aligned: "
            f"{n_rows} values, {len(bases)} bases, "
            f"{len(concept_budget_keys)} concept keys, {len(families)} families."
        )
    basis_array = np.asarray(list(bases), dtype=object)
    weights = _concept_budget_weights(
        _value_basis_weights(values, basis_array), concept_budget_keys
    )
    unique_bases = sorted(set(basis_array.tolist()))
    if not unique_bases:
        return weights
    basis_total = len(weights) / len(unique_bases)
    for basis in unique_bases:
        mask = basis_array == basis
        current_total = weights[mask].sum()
        if current_total > 0:
            weights[mask] *= basis_total / current_total
    weights = weights / weights.mean()
    if not family_multipliers:
        return weights
    family_array = np.asarray(list(families), dtype=object)
    for family, multiplier in sorted(family_multipliers.items()):
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(
                f"Target family loss multiplier for {family!r} must be positive "
                f"and finite, got {multiplier!r}."
            )
        mask = family_array == family
        if not mask.any():
            raise ValueError(
                f"--target-family-loss-multiplier family {family!r} matches "
                "no compiled target."
            )
        weights[mask] *= multiplier
    return weights / weights.mean()


def _value_basis_weights(values: Sequence[float], bases: np.ndarray) -> np.ndarray:
    """Step 2: ``max(|value|, 1) ** 0.5`` over its basis mean."""

    weights = np.ones(len(bases), dtype=np.float64)
    magnitudes = np.asarray(
        [max(abs(float(value)), 1.0) for value in values],
        dtype=np.float64,
    )
    raw_weights = magnitudes**US_FISCAL_TARGET_VALUE_WEIGHT_POWER
    for basis in sorted(set(bases.tolist())):
        mask = bases == basis
        mean_value = raw_weights[mask].mean()
        if mean_value > 0:
            weights[mask] = raw_weights[mask] / mean_value
    return weights


def _concept_budget_weights(
    weights: np.ndarray, concept_budget_keys: Sequence[Hashable]
) -> np.ndarray:
    """Step 3: each group sums to its largest member's weight."""

    for indices in concept_budget_groups(concept_budget_keys):
        group_weights = weights[indices]
        group_total = float(group_weights.sum())
        group_budget = float(group_weights.max(initial=0.0))
        if group_total > 0 and group_budget > 0:
            weights[indices] *= group_budget / group_total
    return weights


def concept_budget_groups(
    concept_budget_keys: Sequence[Hashable],
) -> list[list[int]]:
    """Row indices sharing each concept-budget key, in first-row order."""

    group_indices: dict[Hashable, list[int]] = {}
    for index, key in enumerate(concept_budget_keys):
        group_indices.setdefault(key, []).append(index)
    return list(group_indices.values())


# ---------------------------------------------------------------------------
# Row mappings
# ---------------------------------------------------------------------------


_GroupValidator = Callable[[Sequence[Any], Sequence[str], Sequence[Hashable]], None]


@dataclass(frozen=True)
class TargetLossRowMapping:
    """How one calibration tells the formula what each of its targets is.

    ``value_basis`` returns ``"count"`` or ``"amount"`` and
    ``concept_budget_key`` a hashable key; both read one target spec.
    ``validate``, if set, sees the whole surface with its bases and keys and
    raises if the grouping is not the one the mapping promises.
    """

    mapping_id: str
    value_basis: Callable[[Any], str]
    concept_budget_key: Callable[[Any], Hashable]
    validate: _GroupValidator | None = None


def fiscal_target_value_basis(spec) -> str:
    metadata = spec.metadata
    measure_mode = metadata.get("measure_mode", "")
    source_measure_id = metadata.get("source_measure_id", "")
    if measure_mode in {
        "indicator_sum",
        "less_than_indicator_sum",
    }:
        return "count"
    if "enrollment" in source_measure_id or "recipients" in source_measure_id:
        return "count"
    if "return" in source_measure_id and "count" in source_measure_id:
        return "count"
    return "amount"


def fiscal_target_concept_budget_key(spec) -> tuple[object, ...]:
    return _ledger_concept_budget_key(
        spec,
        fiscal_target_value_basis(spec),
        US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS,
    )


def _ledger_concept_budget_key(
    spec, basis: str, metadata_exclusions: frozenset[str]
) -> tuple[object, ...]:
    metadata = spec.metadata
    if metadata.get("ledger_geography_level") != "congressional_district":
        return (
            basis,
            spec.entity,
            spec.period,
            spec.family,
            spec.name,
        )
    semantic_metadata = tuple(
        sorted(
            (key, value)
            for key, value in metadata.items()
            if key not in metadata_exclusions
        )
    )
    return (
        basis,
        spec.entity,
        spec.period,
        spec.family,
        spec.filter or "",
        metadata.get("state_fips", ""),
        semantic_metadata,
    )


#: The national release's mapping: the ledger compiler's metadata, as is.
US_FISCAL_TARGET_LOSS_ROW_MAPPING = TargetLossRowMapping(
    mapping_id="us_fiscal_ledger_metadata.v1",
    value_basis=fiscal_target_value_basis,
    concept_budget_key=fiscal_target_concept_budget_key,
)


#: Family of the ACS local release's Census ladder population rows.
ACS_LOCAL_POPULATION_FAMILY = "census_population"
#: ``population_target_specs`` in ``tools/build_us_acs_local_release.py``
#: names ladder population rows by geography, with the FIPS digits after the
#: prefix: two for a state, four (state then district) for a district.
ACS_LOCAL_POPULATION_PREFIXES: Mapping[str, tuple[str, int]] = {
    STATE: ("pop_state_", 2),
    CONGRESSIONAL_DISTRICT: ("pop_cd_", 4),
}
#: Per-row metadata that differs between the districts of one concept in one
#: state, so the ACS local concept key must not read it:
#:
#: - ``ledger_fact_label`` and ``ledger_layout_groupby_value_label``: the
#:   human-readable labels the ledger compiler (``ledger_targets.py``) stamps
#:   on every row, which name the district ("WV congressional district 1").
#:   :data:`US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS` does not list them,
#:   so under the national mapping every compiled district row is its own
#:   group; the national mapping is kept as it is here.
#: - ``state_cd_cd_file_value``: the district file's own value of the row,
#:   which the ``state_cd`` rebase (``tools/us_acs_local_cd_surface.py``)
#:   stamps before rescaling it. The rebase's block-level stamps (parent,
#:   basis, factor, file state sum) are equal across a block and may stay.
#:
#: :func:`validate_acs_local_concept_groups` refuses a surface where any other
#: per-district key splits a reconciliation block.
ACS_LOCAL_PER_DISTRICT_METADATA = frozenset(
    {
        "ledger_fact_label",
        "ledger_layout_groupby_value_label",
        "state_cd_cd_file_value",
    }
)
US_ACS_LOCAL_CONCEPT_METADATA_EXCLUSIONS = (
    US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS | ACS_LOCAL_PER_DISTRICT_METADATA
)
#: The ledger's unit for a measure, and the basis that unit implies.
LEDGER_MEASURE_UNIT_BASIS: Mapping[str, str] = {
    "count": COUNT_BASIS,
    "usd": AMOUNT_BASIS,
}
#: The metadata the national basis rule reads; a ledger row lacking either
#: would fall to "amount" by default, so the ACS local mapping refuses it.
_LEDGER_BASIS_KEYS = ("measure_mode", "source_measure_id")


def acs_local_population_geography(spec) -> tuple[str, str] | None:
    """``(level, fips)`` of a Census ladder population row, else None.

    A ladder row is a ``census_population`` spec with no ledger metadata. Its
    ``geography_level`` must be state or congressional district, and its name
    must carry that level's prefix and FIPS digits; any other combination is
    refused rather than guessed.
    """

    metadata = spec.metadata
    if spec.family != ACS_LOCAL_POPULATION_FAMILY or "ledger_measure_unit" in (
        metadata
    ):
        return None
    level = metadata.get("geography_level")
    if level not in ACS_LOCAL_POPULATION_PREFIXES:
        raise ValueError(
            f"{spec.name}: a Census ladder population row must have "
            f"geography_level state or congressional_district, got {level!r}."
        )
    prefix, n_digits = ACS_LOCAL_POPULATION_PREFIXES[level]
    fips = spec.name.removeprefix(prefix)
    if (
        not spec.name.startswith(prefix)
        or len(fips) != n_digits
        or not (fips.isdigit())
    ):
        raise ValueError(
            f"{spec.name}: a {level} ladder population row is named "
            f"{prefix}<{n_digits} FIPS digits>."
        )
    return level, fips


def acs_local_target_value_basis(spec) -> str:
    """The basis of one ACS local target, classified explicitly."""

    if acs_local_population_geography(spec) is not None:
        return COUNT_BASIS
    metadata = spec.metadata
    unit = metadata.get("ledger_measure_unit")
    if unit is None:
        raise ValueError(
            f"{spec.name} ({spec.family}): the ACS local loss weights classify "
            "ledger-compiled rows and Census ladder population rows only; "
            "this row is neither (it has no ledger_measure_unit)."
        )
    expected = LEDGER_MEASURE_UNIT_BASIS.get(unit)
    if expected is None:
        raise ValueError(
            f"{spec.name}: ledger_measure_unit {unit!r} has no loss basis "
            f"(known: {sorted(LEDGER_MEASURE_UNIT_BASIS)})."
        )
    missing = [key for key in _LEDGER_BASIS_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"{spec.name}: a ledger row needs {missing} to take the national "
            "loss basis."
        )
    basis = fiscal_target_value_basis(spec)
    if basis != expected:
        raise ValueError(
            f"{spec.name}: the national loss basis says {basis!r} but its "
            f"ledger_measure_unit {unit!r} says {expected!r}."
        )
    return basis


def acs_local_target_concept_budget_key(spec) -> tuple[object, ...]:
    """The concept key of one ACS local target, classified explicitly."""

    basis = acs_local_target_value_basis(spec)
    geography = acs_local_population_geography(spec)
    if geography is None:
        if spec.metadata.get(
            "ledger_geography_level"
        ) == CONGRESSIONAL_DISTRICT and not spec.metadata.get("state_fips"):
            raise ValueError(
                f"{spec.name}: a district row without state_fips would share a "
                "concept budget across states."
            )
        return _ledger_concept_budget_key(
            spec, basis, US_ACS_LOCAL_CONCEPT_METADATA_EXCLUSIONS
        )
    level, fips = geography
    if level == STATE:
        return (basis, spec.entity, spec.period, spec.family, spec.name)
    # The national district form: one budget per (concept, state).
    return (
        basis,
        spec.entity,
        spec.period,
        spec.family,
        spec.filter or "",
        fips[:2],
        (("geography_level", CONGRESSIONAL_DISTRICT),),
    )


def validate_acs_local_concept_groups(
    specs: Sequence[Any],
    bases: Sequence[str],
    concept_budget_keys: Sequence[Hashable],
) -> None:
    """Refuse an ACS local grouping that is not the promised one.

    Each ``state_cd`` reconciliation block (the district rows sharing one
    ``state_cd_parent_target_name``) must be exactly one concept group: a
    per-district metadata key leaking into the key would split it into one
    group per district, and two blocks sharing a key would pool budgets.
    Each district group must lie in one state.
    """

    del bases
    group_of_parent: dict[str, set[Hashable]] = {}
    parent_of_group: dict[Hashable, set[str]] = {}
    states_of_group: dict[Hashable, set[str]] = {}
    for spec, key in zip(specs, concept_budget_keys, strict=True):
        population = acs_local_population_geography(spec)
        if population is not None:
            level, fips = population
            if level == CONGRESSIONAL_DISTRICT:
                states_of_group.setdefault(key, set()).add(fips[:2])
            continue
        metadata = spec.metadata
        if metadata.get("ledger_geography_level") != CONGRESSIONAL_DISTRICT:
            continue
        states_of_group.setdefault(key, set()).add(str(metadata.get("state_fips")))
        parent = metadata.get("state_cd_parent_target_name")
        if parent:
            group_of_parent.setdefault(str(parent), set()).add(key)
            parent_of_group.setdefault(key, set()).add(str(parent))
    split = {parent: keys for parent, keys in group_of_parent.items() if len(keys) > 1}
    if split:
        parent, keys = next(iter(sorted(split.items())))
        raise ValueError(
            f"{len(split)} state_cd block(s) split across concept groups, e.g. "
            f"the district rows under {parent} land in {len(keys)} groups; a "
            "per-district metadata key is reaching the concept key."
        )
    pooled = {
        key: parents for key, parents in parent_of_group.items() if len(parents) > 1
    }
    if pooled:
        parents = next(iter(pooled.values()))
        raise ValueError(
            f"{len(pooled)} concept group(s) pool several state_cd blocks, e.g. "
            f"{sorted(parents)[:3]}."
        )
    spread = [states for states in states_of_group.values() if len(states) > 1]
    if spread:
        raise ValueError(
            f"{len(spread)} district concept group(s) span states, e.g. "
            f"{sorted(spread[0])}."
        )


#: The ACS local-area release's mapping (see the module docstring).
US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING = TargetLossRowMapping(
    mapping_id="us_acs_local.v1",
    value_basis=acs_local_target_value_basis,
    concept_budget_key=acs_local_target_concept_budget_key,
    validate=validate_acs_local_concept_groups,
)


# ---------------------------------------------------------------------------
# Spec-level entry points
# ---------------------------------------------------------------------------


def target_loss_weights(
    specs: Iterable[Any],
    family_multipliers: Mapping[str, float] | None = None,
    *,
    row_mapping: TargetLossRowMapping = US_FISCAL_TARGET_LOSS_ROW_MAPPING,
) -> np.ndarray:
    """The formula's weights for ``specs``, row-aligned, under one mapping."""

    specs = tuple(specs)
    bases = [row_mapping.value_basis(spec) for spec in specs]
    keys = [row_mapping.concept_budget_key(spec) for spec in specs]
    if row_mapping.validate is not None:
        row_mapping.validate(specs, bases, keys)
    return target_loss_weights_from_rows(
        [spec.value for spec in specs],
        bases,
        keys,
        [spec.family for spec in specs],
        family_multipliers,
    )


def fiscal_target_loss_weights(
    registry,
    family_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    """The national release's weights for a compiled ``TargetRegistry``."""

    return target_loss_weights(registry.specs, family_multipliers)


def fiscal_target_value_basis_weights(registry) -> np.ndarray:
    """Step 2 alone, under the national mapping."""

    bases = np.asarray(
        [fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )
    return _value_basis_weights([spec.value for spec in registry.specs], bases)


def fiscal_target_concept_budget_weights(registry) -> np.ndarray:
    """Steps 2-3 alone, under the national mapping."""

    return _concept_budget_weights(
        fiscal_target_value_basis_weights(registry),
        [fiscal_target_concept_budget_key(spec) for spec in registry.specs],
    )


def us_acs_local_target_loss_weights(
    specs: Iterable[Any],
    family_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    """The ACS local release's weights for its training target specs."""

    return target_loss_weights(
        specs,
        family_multipliers,
        row_mapping=US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING,
    )


# ---------------------------------------------------------------------------
# Options, digests and summaries
# ---------------------------------------------------------------------------


def parse_target_family_loss_multipliers(entries: Iterable[str]) -> dict[str, float]:
    """``FAMILY=MULTIPLIER`` command-line entries as a mapping.

    Raises ``ValueError`` for a malformed entry, a multiplier that is not
    positive and finite, or a repeated family.
    """

    multipliers: dict[str, float] = {}
    for entry in entries:
        family, separator, raw_value = entry.partition("=")
        try:
            value = float(raw_value)
        except ValueError:
            value = math.nan
        if not separator or not family or not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "--target-family-loss-multiplier expects FAMILY=MULTIPLIER "
                f"with a positive finite multiplier, got {entry!r}."
            )
        if family in multipliers:
            raise ValueError(
                f"--target-family-loss-multiplier repeats family {family!r}."
            )
        multipliers[family] = value
    return multipliers


def target_row_name(spec) -> str:
    """A spec's diagnostic row name, ``name@period``."""

    return f"{spec.name}@{spec.period}"


def target_loss_weights_sha256(names: Sequence[str], weights: np.ndarray) -> str:
    """Content digest of a row-aligned loss-weight vector.

    The canonical form is the national release's ``loss_vector_sha256``: one
    ``{"row_name", "weight_hex"}`` object per row (row names as
    :func:`target_row_name` gives them), in row order, as compact sorted-key
    JSON. Any change to a name, a weight bit or the row order changes it.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(names),):
        raise ValueError(
            f"{weights.shape} loss weights do not align with {len(names)} names."
        )
    rows = [
        {"row_name": str(name), "weight_hex": float(weight).hex()}
        for name, weight in zip(names, weights, strict=True)
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def target_geography_level(spec) -> str:
    """A spec's geography level for summaries (ledger or ladder metadata)."""

    metadata = spec.metadata
    return str(
        metadata.get("ledger_geography_level")
        or metadata.get("geography_level")
        or "unspecified"
    )


def target_loss_weight_distribution(
    specs: Sequence[Any],
    weights: np.ndarray,
    *,
    row_mapping: TargetLossRowMapping,
) -> dict[str, object]:
    """Loss-weight shares by family x geography level x basis, and by basis.

    ``loss_share`` is a cell's share of the total weight (the share of the
    loss it would carry at equal misses); ``equal_share`` is what it would
    carry with every target weighted equally. District concept groups are
    counted too.
    """

    specs = tuple(specs)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(specs),):
        raise ValueError(
            f"{weights.shape} loss weights do not align with {len(specs)} specs."
        )
    bases = [row_mapping.value_basis(spec) for spec in specs]
    keys = [row_mapping.concept_budget_key(spec) for spec in specs]
    total = float(weights.sum())
    n_rows = len(specs)

    def cell(mask: np.ndarray) -> dict[str, object]:
        selected = weights[mask]
        return {
            "n_targets": int(mask.sum()),
            "weight_sum": float(selected.sum()),
            "loss_share": float(selected.sum()) / total if total else 0.0,
            "equal_share": float(mask.sum()) / n_rows if n_rows else 0.0,
            "min_weight": float(selected.min()) if selected.size else None,
            "max_weight": float(selected.max()) if selected.size else None,
        }

    families = np.asarray([str(spec.family) for spec in specs], dtype=object)
    levels = np.asarray([target_geography_level(spec) for spec in specs], dtype=object)
    basis_array = np.asarray(bases, dtype=object)
    cells = []
    for family, level, basis in sorted(
        set(zip(families.tolist(), levels.tolist(), bases, strict=True))
    ):
        mask = (families == family) & (levels == level) & (basis_array == basis)
        cells.append(
            {"family": family, "geography_level": level, "basis": basis, **cell(mask)}
        )
    district_rows = levels == CONGRESSIONAL_DISTRICT
    groups = concept_budget_groups(keys)
    district_groups = [
        indices for indices in groups if all(district_rows[i] for i in indices)
    ]
    return {
        "row_mapping": row_mapping.mapping_id,
        "n_targets": n_rows,
        "by_family_level_basis": cells,
        "by_basis": {basis: cell(basis_array == basis) for basis in sorted(set(bases))},
        "concept_groups": {
            "n_groups": len(groups),
            "n_district_rows": int(district_rows.sum()),
            "n_district_groups": len(district_groups),
            "n_singleton_district_groups": sum(
                1 for indices in district_groups if len(indices) == 1
            ),
            "max_district_group_size": max(
                (len(indices) for indices in district_groups), default=0
            ),
        },
    }
