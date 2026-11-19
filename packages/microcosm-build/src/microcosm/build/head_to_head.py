"""Compare a candidate with a reference artifact on one scored target surface.

Every country scores each artifact on one compiled target surface with
``microcosm.calibrate``'s canonical loss (``score_targets`` /
``relative_error_loss``). This module compares the two scored results, and
nothing else: for each target, whose absolute relative error is lower; how
many targets each side has lower; the loss gap; and the same counts within
each target family. Loading and scoring stay with each country (the US
tool's pools, household slices and by-origin battery; the UK's frame, weight
year and engine measures).

The comparison decides nothing. It is evidence for the owner's decision to
promote a build, with no threshold applied: the US head-to-head's rule from
its start, and the UK cut's since microcosm#1095 retired its strict rule-1
verdict against the enhanced FRS. The US tool keeps its own payload's names
("incumbent") for its existing scorecards; the UK receipt uses this module's
("reference").
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HEAD_TO_HEAD_SCHEMA_VERSION",
    "LOWER_CANDIDATE",
    "LOWER_EQUAL",
    "LOWER_REFERENCE",
    "HeadToHead",
    "ScoredTarget",
    "TargetComparison",
    "compare_scored_targets",
    "lower_side",
    "scored_targets_from_mapping",
]

HEAD_TO_HEAD_SCHEMA_VERSION = 1
LOWER_CANDIDATE = "candidate"
LOWER_REFERENCE = "reference"
LOWER_EQUAL = "equal"


@dataclass(frozen=True)
class ScoredTarget:
    """One target's signed relative error under one artifact."""

    name: str
    relative_error: float
    period: Hashable | None = None
    family: str | None = None


@dataclass(frozen=True)
class TargetComparison:
    """One target, scored under both artifacts."""

    name: str
    period: Hashable | None
    family: str | None
    candidate_relative_error: float
    reference_relative_error: float
    lower: str

    @property
    def candidate_absolute_relative_error(self) -> float:
        return abs(self.candidate_relative_error)

    @property
    def reference_absolute_relative_error(self) -> float:
        return abs(self.reference_relative_error)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "period": self.period,
            "family": self.family,
            "candidate_relative_error": self.candidate_relative_error,
            "reference_relative_error": self.reference_relative_error,
            "candidate_absolute_relative_error": (
                self.candidate_absolute_relative_error
            ),
            "reference_absolute_relative_error": (
                self.reference_absolute_relative_error
            ),
            "candidate_minus_reference_absolute_relative_error": (
                self.candidate_absolute_relative_error
                - self.reference_absolute_relative_error
            ),
            "lower_absolute_relative_error": self.lower,
        }


def lower_side(candidate_error: float, reference_error: float) -> str:
    """Which side's absolute relative error is lower; ``equal`` on a tie.

    Signed errors are compared on magnitude. A comparison involving NaN is
    neither lower, so it counts as equal, as both countries' scorers did.
    """

    candidate = abs(float(candidate_error))
    reference = abs(float(reference_error))
    if candidate < reference:
        return LOWER_CANDIDATE
    if candidate > reference:
        return LOWER_REFERENCE
    return LOWER_EQUAL


def _counts(comparisons: Sequence[TargetComparison]) -> dict[str, int]:
    counts = {LOWER_CANDIDATE: 0, LOWER_REFERENCE: 0, LOWER_EQUAL: 0}
    for comparison in comparisons:
        counts[comparison.lower] += 1
    return {
        "candidate_lower_count": counts[LOWER_CANDIDATE],
        "reference_lower_count": counts[LOWER_REFERENCE],
        "equal_count": counts[LOWER_EQUAL],
    }


@dataclass(frozen=True)
class HeadToHead:
    """Two artifacts compared target by target, with their losses."""

    candidate_loss: float
    reference_loss: float
    comparisons: tuple[TargetComparison, ...]

    @property
    def counts(self) -> dict[str, int]:
        return _counts(self.comparisons)

    def counts_by_family(self) -> dict[str, dict[str, int]]:
        """The counts within each family, families sorted; ``None`` is kept."""

        grouped: dict[str | None, list[TargetComparison]] = {}
        for comparison in self.comparisons:
            grouped.setdefault(comparison.family, []).append(comparison)
        return {
            family: _counts(rows)
            for family, rows in sorted(
                grouped.items(), key=lambda item: (item[0] is None, item[0] or "")
            )
        }

    def to_payload(self) -> dict[str, Any]:
        """The versioned, country-neutral comparison block, with no verdict."""

        return {
            "schema_version": HEAD_TO_HEAD_SCHEMA_VERSION,
            "decision": "owner_decides",
            "no_threshold_applied": True,
            "loss": {
                "candidate": self.candidate_loss,
                "reference": self.reference_loss,
                "candidate_minus_reference": self.candidate_loss - self.reference_loss,
            },
            "targets": {
                **self.counts,
                "compared": [
                    comparison.to_payload() for comparison in self.comparisons
                ],
            },
            "by_family": self.counts_by_family(),
        }


def compare_scored_targets(
    candidate: Sequence[ScoredTarget],
    reference: Sequence[ScoredTarget],
    *,
    candidate_loss: float,
    reference_loss: float,
) -> HeadToHead:
    """Compare two scorings of the same target surface, in candidate order.

    Both sequences must name the same targets in the same order (name and
    period). A target's family comes from the candidate's scoring, and must
    not disagree with the reference's where both give one.

    Raises:
        ValueError: If the two scorings cover different targets.
    """

    if len(candidate) != len(reference):
        raise ValueError(
            "The candidate and reference scorings cover different numbers of "
            f"targets ({len(candidate)} and {len(reference)})."
        )
    comparisons = []
    for candidate_row, reference_row in zip(candidate, reference, strict=True):
        key = (candidate_row.name, candidate_row.period)
        reference_key = (reference_row.name, reference_row.period)
        if key != reference_key:
            raise ValueError(
                f"Target comparison contracts differ: {reference_key!r} vs {key!r}."
            )
        if (
            candidate_row.family is not None
            and reference_row.family is not None
            and candidate_row.family != reference_row.family
        ):
            raise ValueError(
                f"Target {key!r} belongs to family {candidate_row.family!r} in the "
                f"candidate's scoring but {reference_row.family!r} in the "
                "reference's."
            )
        comparisons.append(
            TargetComparison(
                name=candidate_row.name,
                period=candidate_row.period,
                family=(
                    candidate_row.family
                    if candidate_row.family is not None
                    else reference_row.family
                ),
                candidate_relative_error=float(candidate_row.relative_error),
                reference_relative_error=float(reference_row.relative_error),
                lower=lower_side(
                    candidate_row.relative_error, reference_row.relative_error
                ),
            )
        )
    return HeadToHead(
        candidate_loss=float(candidate_loss),
        reference_loss=float(reference_loss),
        comparisons=tuple(comparisons),
    )


def scored_targets_from_mapping(
    errors: Mapping[str, float],
    *,
    families: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
) -> list[ScoredTarget]:
    """Scored targets from ``{name: signed relative error}``, in ``order``.

    ``order`` defaults to the names sorted; every name in it must be scored.
    """

    names = sorted(errors) if order is None else list(order)
    missing = [name for name in names if name not in errors]
    if missing:
        raise ValueError(f"No score for targets {missing[:10]}.")
    families = {} if families is None else families
    return [
        ScoredTarget(
            name=name, relative_error=float(errors[name]), family=families.get(name)
        )
        for name in names
    ]
