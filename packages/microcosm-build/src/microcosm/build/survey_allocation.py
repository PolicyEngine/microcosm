"""Domain allocation of supplied design weights, before support cloning.

This numerical primitive performs no source classification, subsampling, frame
mutation or graph activation. Callers must supply an aligned household axis and
reviewed domain declarations. A row can be a physical housing unit or a cluster
of people; explicit unit counts keep its statistical grain out of its weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from types import MappingProxyType

import numpy as np

from microcosm.frame import WeightKind, Weights


class SurveyAllocationError(ValueError):
    """A supplied allocation contract cannot represent its household axis."""


def _label(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SurveyAllocationError(f"{field} must be a nonempty string")
    return value


@dataclass(frozen=True)
class SurveyDomain:
    """One declared population/period and its prespecified source shares.

    Shares must be exact Fractions summing to one. Unit and basis labels are
    declarations, not evidence of survey equivalence or source authentication.
    """

    key: str
    statistical_unit: str
    period: str
    basis: str
    source_shares: Mapping[str, Fraction]

    def __post_init__(self) -> None:
        for field in ("key", "statistical_unit", "period", "basis"):
            _label(getattr(self, field), field)
        if not isinstance(self.source_shares, Mapping) or not self.source_shares:
            raise SurveyAllocationError("source_shares must be a nonempty mapping")
        shares = {}
        for source, share in self.source_shares.items():
            _label(source, "source channel")
            if not isinstance(share, Fraction) or not 0 <= share <= 1:
                raise SurveyAllocationError(
                    "shares must be Fractions between zero and one"
                )
            shares[source] = share
        if sum(shares.values(), Fraction()) != 1:
            raise SurveyAllocationError("domain shares must sum exactly to one")
        object.__setattr__(
            self, "source_shares", MappingProxyType(dict(sorted(shares.items())))
        )


@dataclass(frozen=True)
class DomainSourceEstimate:
    """Separate estimates in one declared unit, period and source domain."""

    domain: str
    source: str
    statistical_unit: str
    period: str
    basis: str
    source_share: Fraction
    rows: int
    design_estimate: float
    allocated_estimate: float


@dataclass(frozen=True)
class SurveyAllocation:
    """Output weights aligned to input row IDs; estimates never pool units."""

    row_ids: tuple[int, ...]
    weights: Weights
    estimates: tuple[DomainSourceEstimate, ...]


def _axis(values: Sequence, size: int, field: str) -> tuple:
    if isinstance(values, (str, bytes)):
        raise SurveyAllocationError(f"{field} must be a row-aligned sequence")
    result = tuple(values)
    if len(result) != size:
        raise SurveyAllocationError(f"{field} must match the household weight axis")
    return result


def allocate_domain_weights(
    design_weights: Weights,
    *,
    row_ids: Sequence[int],
    source_channels: Sequence[str],
    domain_keys: Sequence[str | None],
    unit_counts: Sequence[int],
    domains: Sequence[SurveyDomain],
) -> SurveyAllocation:
    """Multiply each household weight by its declared domain/source share.

    ``unit_counts`` is mandatory: one for a housing unit or individual GQ
    placeholder, the number of covered people for a person-counting cluster.
    It affects accounting, never the household weight multiplier. No source
    total is normalized to another source. Unknown domains and positive-share
    cells lacking positive design support refuse. Known exclusions must be
    selected explicitly before this operation; rows are never dropped here.

    The declarations do not authenticate their inputs. A production graph
    adapter must bind IDs, source membership, coverage and period assumptions
    to the actual population and use a new allocation context/receipt version.
    """
    if (
        not isinstance(design_weights, Weights)
        or design_weights.kind is not WeightKind.DESIGN
    ):
        raise SurveyAllocationError("allocation requires DESIGN weights")
    size = len(design_weights.values)
    ids = _axis(row_ids, size, "row_ids")
    if any(
        isinstance(x, bool) or not isinstance(x, Integral) or not 0 < x <= 2**63 - 1
        for x in ids
    ):
        raise SurveyAllocationError(
            "row_ids must be positive int64 household identities"
        )
    if len(set(ids)) != size:
        raise SurveyAllocationError("household identities must be unique")
    sources = _axis(source_channels, size, "source_channels")
    keys = _axis(domain_keys, size, "domain_keys")
    counts = _axis(unit_counts, size, "unit_counts")
    if any(
        isinstance(x, bool) or not isinstance(x, Integral) or not 0 < x <= 2**53
        for x in counts
    ):
        raise SurveyAllocationError(
            "unit_counts must be positive exactly representable integers"
        )
    declared = {}
    for domain in domains:
        if not isinstance(domain, SurveyDomain) or domain.key in declared:
            raise SurveyAllocationError(
                "domains must be unique SurveyDomain declarations"
            )
        declared[domain.key] = domain
    cells: dict[tuple[str, str], list[int]] = {}
    for position, (key, source) in enumerate(zip(keys, sources, strict=True)):
        _label(source, "source channel")
        if not isinstance(key, str) or key not in declared:
            raise SurveyAllocationError("unknown or undeclared coverage domain")
        if source not in declared[key].source_shares:
            raise SurveyAllocationError(
                "source is not declared for its coverage domain"
            )
        cells.setdefault((key, source), []).append(position)
    incoming = design_weights.values
    allocated = np.zeros(size, dtype=np.float64)
    unit_array = np.asarray(counts, dtype=np.float64)
    estimates = []
    for key, domain in sorted(declared.items()):
        for source, share in domain.source_shares.items():
            positions = np.asarray(cells.get((key, source), ()), dtype=np.int64)
            values = incoming[positions]
            if share > 0 and not np.any(values > 0):
                raise SurveyAllocationError(
                    "positive-share domain/source lacks sampled support"
                )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                output = values * float(share)
                design_total = float(np.sum(values * unit_array[positions]))
                allocated_total = float(np.sum(output * unit_array[positions]))
            if not np.isfinite(design_total) or not np.isfinite(allocated_total):
                raise SurveyAllocationError("domain estimate overflow")
            if share > 0 and np.any((values > 0) & (output == 0)):
                raise SurveyAllocationError("positive allocated weight underflow")
            allocated[positions] = output
            estimates.append(
                DomainSourceEstimate(
                    key,
                    source,
                    domain.statistical_unit,
                    domain.period,
                    domain.basis,
                    share,
                    len(positions),
                    design_total,
                    allocated_total,
                )
            )
    return SurveyAllocation(
        tuple(int(x) for x in ids),
        Weights(allocated, WeightKind.IMPORTANCE),
        tuple(estimates),
    )
