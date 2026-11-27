"""The release files a spec country's build adds to the base three.

Placeholder for tracker 5b · 13: the payload builders are written in the
next commit.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "demographics_payload",
    "reform_validation_payload",
    "source_coverage_payload",
]


def source_coverage_payload(
    *,
    country: str,
    target_references: Mapping,
    holdout_references: Mapping | None,
    diagnostics: Mapping,
    source_contract: Mapping[str, object],
) -> dict[str, object]:
    return {}


def demographics_payload(
    *,
    country: str,
    target_references: Mapping,
    diagnostics: Mapping,
    period: int,
) -> dict[str, object]:
    return {}


def reform_validation_payload(*, country: str, period: int) -> dict[str, object]:
    return {}
