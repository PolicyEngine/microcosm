"""Whole-household domain sampling composed with survey-share allocation.

Coverage columns are supplied declarations bound to this Frame's household
axis. This module does not authenticate their survey meaning, align periods,
classify sources, or authorize a production graph. It never normalizes a
sampled domain estimate to another estimate or to a mixed-unit weight sum.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from microcosm.build.survey_allocation import (
    DomainSourceEstimate,
    SurveyAllocationError,
    SurveyDomain,
    allocate_domain_weights,
)
from microcosm.frame import Frame, MassChange, WeightKind, Weights

PROTOCOL = "microcosm.survey-domain-sample-allocation.v1"


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _id_digest(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


@dataclass(frozen=True)
class DomainHouseholdSelection:
    """Positions drawn from one complete, caller-supplied source/domain cell.

    This arithmetic result does not authenticate the catalogue or its coverage.
    Positions always refer to the original input axis, before ID remapping.
    """

    domain: str
    source: str
    eligible_households: int
    positions: tuple[int, ...]
    inclusion_probability: Fraction | None


def select_domain_households(
    *,
    row_ids: Sequence[int] | Sequence[str],
    source_channels: Sequence[str],
    domain_keys: Sequence[str],
    cells: Sequence[tuple[str, str]],
    fraction: Fraction,
    seed: int,
) -> tuple[DomainHouseholdSelection, ...]:
    """Use the same per-cell draw for native catalogues and full Frames.

    ``cells`` contains (domain, source) pairs, including declared empty cells.
    IDs must be uniformly positive int64 integers or nonempty literal strings.
    They are unique within each source, and sort numerically or lexically,
    respectively. No ID conversion, source-share exclusion, classification,
    weighting, or claim of full-source completeness occurs here. A source owner
    must establish the complete eligible axis before calling this function.

    Each populated cell retains max(1, floor(fraction * N)) rows. Empty cells
    have no inclusion probability; full selection consumes no random draw.
    The existing protocol and PCG64 seed construction remain unchanged.
    """
    if not isinstance(fraction, Fraction) or not 0 < fraction <= 1:
        raise SurveyAllocationError("fraction must be a Fraction in (0, 1]")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise SurveyAllocationError("seed must be an unsigned 64-bit integer")
    axes = (row_ids, source_channels, domain_keys, cells)
    if any(isinstance(axis, (str, bytes)) for axis in axes):
        raise SurveyAllocationError("selection inputs must be row sequences")
    ids, sources, domains, declared = (tuple(axis) for axis in axes)
    if len(ids) != len(sources) or len(ids) != len(domains):
        raise SurveyAllocationError("selection axes must have equal lengths")
    integer_ids = all(
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and 0 < value <= 2**63 - 1
        for value in ids
    )
    string_ids = all(type(value) is str and bool(value.strip()) for value in ids)
    if not integer_ids and not string_ids:
        raise SurveyAllocationError(
            "selection IDs must be uniformly positive int64 or literal strings"
        )
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (*sources, *domains)
    ):
        raise SurveyAllocationError("selection source/domain labels must be nonempty")
    if any(
        type(cell) is not tuple
        or len(cell) != 2
        or any(not isinstance(label, str) or not label.strip() for label in cell)
        for cell in declared
    ):
        raise SurveyAllocationError("selection cells require (domain, source) pairs")
    if len(set(declared)) != len(declared):
        raise SurveyAllocationError("selection cells must be unique")
    if len(set(zip(sources, ids, strict=True))) != len(ids):
        raise SurveyAllocationError("selection identities must be unique within source")
    groups: dict[tuple[str, str], list[int]] = {cell: [] for cell in declared}
    for position, cell in enumerate(zip(domains, sources, strict=True)):
        if cell not in groups:
            raise SurveyAllocationError("selection row has an undeclared cell")
        groups[cell].append(position)
    result = []
    for (domain, source), group in sorted(groups.items()):
        positions = np.asarray(sorted(group, key=ids.__getitem__), dtype=np.int64)
        eligible = len(positions)
        if not eligible:
            chosen, probability = (), None
        else:
            count = max(1, fraction.numerator * eligible // fraction.denominator)
            probability = Fraction(count, eligible)
            if count == eligible:
                chosen = positions
            else:
                material = _json([PROTOCOL, seed, source, domain])
                cell_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
                rng = np.random.Generator(np.random.PCG64(cell_seed))
                chosen = np.sort(rng.choice(positions, size=count, replace=False))
        result.append(
            DomainHouseholdSelection(
                domain, source, eligible, tuple(int(p) for p in chosen), probability
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class SampledDomainCell:
    """One source/domain estimate in its declared unit, period and basis."""

    full: DomainSourceEstimate
    selected_households: int
    inclusion_probability: Fraction | None
    selected_household_ids_sha256: str
    sampled_allocated_estimate: float


@dataclass(frozen=True)
class DomainSampleAllocation:
    """A sampled IMPORTANCE Frame and separate, immutable domain accounting."""

    frame: Frame
    cells: tuple[SampledDomainCell, ...]
    fraction: Fraction
    seed: int
    input_axis_sha256: str
    domain_declarations_sha256: str
    numpy_version: str
    coverage_columns: tuple[str, str, str]

    def receipt(self) -> dict[str, object]:
        """Describe the executed arithmetic without claiming source admission."""
        return {
            "protocol": PROTOCOL,
            "coverage_status": "declared_frame_columns_only",
            "release_eligible": False,
            "coverage_columns": dict(
                zip(
                    ("source", "domain", "unit_count"),
                    self.coverage_columns,
                    strict=True,
                )
            ),
            "fraction": [self.fraction.numerator, self.fraction.denominator],
            "seed": self.seed,
            "rng": {"family": "PCG64", "numpy_version": self.numpy_version},
            "input_axis_sha256": self.input_axis_sha256,
            "domain_declarations_sha256": self.domain_declarations_sha256,
            "cells": [
                {
                    "domain": cell.full.domain,
                    "source": cell.full.source,
                    "statistical_unit": cell.full.statistical_unit,
                    "period": cell.full.period,
                    "basis": cell.full.basis,
                    "source_share": [
                        cell.full.source_share.numerator,
                        cell.full.source_share.denominator,
                    ],
                    "eligible_households": cell.full.rows,
                    "selected_households": cell.selected_households,
                    "inclusion_probability": (
                        None
                        if cell.inclusion_probability is None
                        else [
                            cell.inclusion_probability.numerator,
                            cell.inclusion_probability.denominator,
                        ]
                    ),
                    "selected_household_ids_sha256": cell.selected_household_ids_sha256,
                    "full_design_estimate": cell.full.design_estimate,
                    "full_allocated_estimate": cell.full.allocated_estimate,
                    "sampled_allocated_estimate": cell.sampled_allocated_estimate,
                }
                for cell in self.cells
            ],
        }


def sample_and_allocate_domains(
    frame: Frame,
    *,
    source_column: str,
    domain_column: str,
    unit_count_column: str,
    domains: Sequence[SurveyDomain],
    fraction: Fraction,
    seed: int,
) -> DomainSampleAllocation:
    """Sample each declared source/domain cell, then compose its multipliers.

    Each populated cell retains ``max(1, floor(fraction * N))`` households by
    simple random sampling without replacement. Its output household weight
    is ``source_design_weight * domain_source_share * N / n``. Unit counts
    affect estimates only. Small domains therefore remain represented, with
    their own actual inclusion probability recorded; no totals are forced to
    match. Full selection does not draw.

    PCG64 streams are separated by protocol, source and domain. Sorted household
    IDs make selection independent of row order and unrelated cells. The
    returned Frame preserves complete selected lineages through Frame.select.
    It never labels the sampling/allocation result DESIGN.

    A positive-share cell whose random sample has no positive weight refuses.
    Callers must not search seeds to evade that refusal. Coverage exclusions and
    period harmonization must be resolved explicitly before this operation.
    """
    if not isinstance(frame, Frame):
        raise TypeError("domain sampling requires a Frame")
    frame.revalidate()
    if frame.weighted_entities != ("household",):
        raise SurveyAllocationError("domain sampling requires household-only weights")
    if not isinstance(fraction, Fraction) or not 0 < fraction <= 1:
        raise SurveyAllocationError("fraction must be a Fraction in (0, 1]")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise SurveyAllocationError("seed must be an unsigned 64-bit integer")
    names = (source_column, domain_column, unit_count_column)
    if any(not isinstance(name, str) or not name for name in names):
        raise SurveyAllocationError("coverage column names must be nonempty strings")
    if len(set(names)) != 3 or "household_id" in names:
        raise SurveyAllocationError("coverage columns must be distinct from identities")
    household = frame.table("household")
    if not set(names).issubset(household):
        raise SurveyAllocationError("household coverage columns are missing")
    ids = household["household_id"].to_numpy()
    sources = tuple(household[source_column])
    keys = tuple(household[domain_column])
    counts = tuple(household[unit_count_column])
    domains = tuple(domains)
    weights = frame.weights_for("household")
    full = allocate_domain_weights(
        weights,
        row_ids=ids,
        source_channels=sources,
        domain_keys=keys,
        unit_counts=counts,
        domains=domains,
    )
    axis = hashlib.sha256()
    for row, source, key, count, weight in zip(
        ids, sources, keys, counts, weights.values, strict=True
    ):
        axis.update(_json([int(row), source, key, int(count), float(weight)]))
        axis.update(b"\n")
    declarations = [
        {
            "key": domain.key,
            "unit": domain.statistical_unit,
            "period": domain.period,
            "basis": domain.basis,
            "shares": {
                source: [share.numerator, share.denominator]
                for source, share in domain.source_shares.items()
            },
        }
        for domain in sorted(domains, key=lambda domain: domain.key)
    ]
    selection = select_domain_households(
        row_ids=ids,
        source_channels=sources,
        domain_keys=keys,
        cells=tuple((estimate.domain, estimate.source) for estimate in full.estimates),
        fraction=fraction,
        seed=seed,
    )
    selected_positions = []
    output_by_id = {}
    cells = []
    for estimate, cell in zip(full.estimates, selection, strict=True):
        chosen = np.asarray(cell.positions, dtype=np.int64)
        selected = len(chosen)
        probability = cell.inclusion_probability
        if selected:
            values = full.weights.values[chosen]
            if estimate.source_share > 0 and not np.any(values > 0):
                raise SurveyAllocationError(
                    "positive-share domain/source lacks sampled positive weight"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                output = values * float(1 / probability)
                sample_estimate = float(
                    np.sum(output * np.asarray([counts[p] for p in chosen]))
                )
            if not np.isfinite(output).all() or not np.isfinite(sample_estimate):
                raise SurveyAllocationError(
                    "sampled domain weight or estimate overflow"
                )
            selected_positions.extend(chosen.tolist())
            output_by_id.update(zip(ids[chosen].tolist(), output.tolist(), strict=True))
            selected_ids = np.sort(ids[chosen])
        else:
            selected, probability, sample_estimate = 0, None, 0.0
            selected_ids = ()
        cells.append(
            SampledDomainCell(
                estimate,
                selected,
                probability,
                _id_digest(selected_ids),
                sample_estimate,
            )
        )
    selected_ids = ids[np.asarray(selected_positions, dtype=np.int64)]
    if len(selected_ids) == len(ids):
        sampled = frame
    else:
        membership = frame.schema.membership_column("household")
        sampled = frame.select(frame.person[membership].isin(selected_ids).to_numpy())
    realized = sampled.table("household")["household_id"].to_numpy()
    if set(realized) != set(selected_ids):
        raise SurveyAllocationError("whole-household sample differs from selection")
    result = sampled.with_weights(
        "household",
        Weights(
            np.asarray([output_by_id[int(row)] for row in realized]),
            WeightKind.IMPORTANCE,
        ),
        mass=MassChange(
            factor=None,
            reason=f"{PROTOCOL}: domain source shares and inverse household inclusion",
        ),
    )
    return DomainSampleAllocation(
        result,
        tuple(cells),
        fraction,
        seed,
        axis.hexdigest(),
        hashlib.sha256(_json(declarations)).hexdigest(),
        np.__version__,
        names,
    )
