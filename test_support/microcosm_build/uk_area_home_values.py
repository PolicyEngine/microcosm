"""Toy knots for the spine's area home values, G8 (microcosm#931).

Invented constituencies and values only, read through the module's own
loader, so a test's resources parse as the committed ones do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from microcosm.build.uk_runtime.area_home_values import (
    UKAreaHomeValuesResources,
    load_area_home_values_resources,
)


def toy_knots_payload(
    seats: Mapping[str, Sequence[float]],
    *,
    grid: Sequence[float],
    cap: float,
    basis_year: int = 2024,
) -> dict[str, object]:
    """A knots resource holding ``seats``' quantiles at ``grid``, at the
    constituency level the area home values read."""

    return {
        "schema_version": 1,
        "kind": "uk_high_value_home_value_knots",
        "grid": list(grid),
        "cap": cap,
        "basis_year": basis_year,
        "inputs_sha256": "toy",
        "method": "toy",
        "levels": {
            "constituency": {
                code: {"quantiles": list(quantiles)}
                for code, quantiles in seats.items()
            }
        },
    }


def toy_area_home_values_resources(
    seats: Mapping[str, Sequence[float]],
    regional_reference: Mapping[str, object],
    *,
    grid: Sequence[float],
    cap: float,
) -> UKAreaHomeValuesResources:
    """The area home values' two resources: ``seats``' toy knots and
    ``regional_reference``, from their JSON bytes."""

    return load_area_home_values_resources(
        knots=json.dumps(toy_knots_payload(seats, grid=grid, cap=cap)).encode(),
        regional_reference=json.dumps(regional_reference).encode(),
    )
