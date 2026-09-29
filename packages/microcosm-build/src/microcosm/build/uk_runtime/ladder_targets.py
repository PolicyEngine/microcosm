"""Diagnostic household sums from the sha-pinned UK OA ladder.

The ladder remains the geography-assignment artifact and stage-one sampling
weight. Its household sums are diagnostics only: calibration targets compile
from the pinned Chronicle feed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.geography_ladder import UkOaLadder

__all__ = [
    "constituency_household_targets",
    "ladder_vs_chronicle_household_dispersion",
    "ladder_target_provenance",
    "local_authority_household_targets",
]

_COUNTRY_BY_CODE_PREFIX = {
    "E": "England",
    "N": "Northern Ireland",
    "S": "Scotland",
    "W": "Wales",
}


def constituency_household_targets(ladder: UkOaLadder) -> pd.DataFrame:
    """Census occupied-household counts by constituency, from the ladder.

    Pair targets and assignment from the SAME loaded ladder: record
    :func:`ladder_target_provenance` (and the ladder artifact's sha) in any
    build manifest so targets from one ladder cannot silently calibrate an
    assignment drawn from another.
    """

    return _household_targets(ladder.constituency_code, ladder)


def local_authority_household_targets(ladder: UkOaLadder) -> pd.DataFrame:
    """Census occupied-household counts by local authority, from the ladder.

    The same pairing discipline as
    :func:`constituency_household_targets` applies.
    """

    return _household_targets(ladder.local_authority_code, ladder)


def ladder_vs_chronicle_household_dispersion(
    ladder: UkOaLadder,
    compiled_specs: Iterable[Any],
) -> dict[str, object]:
    """Compare diagnostic ladder sums with compiled Chronicle household cells."""

    targets_by_level = {
        "constituency": constituency_household_targets(ladder).set_index("code")[
            "households"
        ],
        "local_authority": local_authority_household_targets(ladder).set_index("code")[
            "households"
        ],
    }
    cells: list[dict[str, object]] = []
    for spec in compiled_specs:
        name = str(_spec_field(spec, "name", ""))
        if not name.startswith("ons.census.households@"):
            continue
        metadata = _spec_field(spec, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{name} must carry mapping metadata.")
        level = str(
            metadata.get("ledger_geography_level", metadata.get("geography_level", ""))
        )
        area_code = str(
            metadata.get("ledger_geography_id", metadata.get("geography_id", ""))
        )
        if level not in targets_by_level or not area_code:
            raise ValueError(
                f"{name} must identify a supported Ledger geography level and id."
            )
        ladder_targets = targets_by_level[level]
        if area_code not in ladder_targets.index:
            raise ValueError(
                f"{name} area {area_code!r} is absent from the OA ladder {level}."
            )
        country = _COUNTRY_BY_CODE_PREFIX.get(area_code[:1])
        if country is None:
            raise ValueError(f"{name} has an unrecognised UK area code {area_code!r}.")
        ladder_value = float(ladder_targets.loc[area_code])
        chronicle_value = float(_spec_field(spec, "value", np.nan))
        if not np.isfinite(chronicle_value):
            raise ValueError(f"{name} has a non-finite Ledger household value.")
        cells.append(
            {
                "name": name,
                "geography_level": level,
                "area_code": area_code,
                "country": country,
                "ladder_households": ladder_value,
                "chronicle_households": chronicle_value,
                "delta": ladder_value - chronicle_value,
            }
        )
    if not cells:
        raise ValueError("compiled specs contain no ons.census.households cells.")

    countries: dict[str, dict[str, float | int]] = {}
    for country in sorted({str(cell["country"]) for cell in cells}):
        deltas = np.asarray(
            [cell["delta"] for cell in cells if cell["country"] == country],
            dtype=np.float64,
        )
        countries[country] = _dispersion_summary(deltas)

    ni_constituency_deltas = np.asarray(
        [
            cell["delta"]
            for cell in cells
            if cell["country"] == "Northern Ireland"
            and cell["geography_level"] == "constituency"
        ],
        dtype=np.float64,
    )
    if ni_constituency_deltas.size:
        ni_summary = _dispersion_summary(ni_constituency_deltas)
        if (
            ni_summary["max_absolute_delta"] > 50
            or ni_summary["mean_absolute_delta"] > 15
        ):
            raise ValueError(
                "NI DZ-to-PARLCON24 household dispersion exceeds the publisher "
                "oracle: "
                f"mean absolute delta {ni_summary['mean_absolute_delta']:.3f}, "
                f"max absolute delta {ni_summary['max_absolute_delta']:.3f}."
            )

    return {"countries": countries, "cells": cells}


def _spec_field(spec: Any, name: str, default: Any) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _dispersion_summary(deltas: np.ndarray) -> dict[str, float | int]:
    return {
        "cells": int(deltas.size),
        "mean_absolute_delta": float(np.abs(deltas).mean()),
        "max_absolute_delta": float(np.abs(deltas).max()),
        "net_delta": float(deltas.sum()),
    }


def ladder_target_provenance(ladder: UkOaLadder) -> dict[str, object]:
    """Provenance to record beside ladder-derived targets in a manifest."""

    metadata = ladder.metadata
    return {
        "kind": str(metadata.get("kind", "")),
        "coverage": str(metadata.get("coverage", "")),
        "oa_vintage": str(metadata.get("oa_vintage", "")),
        "constituency_sampling_basis": str(
            metadata.get("constituency_sampling_basis", "")
        ),
        "layer_vintages": dict(ladder.layer_vintages),
        "output_areas": int(len(ladder)),
        "households_total": float(np.asarray(ladder.households).sum()),
    }


def _household_targets(codes: np.ndarray, ladder: UkOaLadder) -> pd.DataFrame:
    households = np.asarray(ladder.households, dtype=np.float64)
    if not np.isfinite(households).all() or (households < 0).any():
        raise ValueError("ladder household counts must be finite and non-negative.")
    code_series = pd.Series(np.asarray(codes, dtype=object))
    if code_series.isna().any():
        raise ValueError(
            f"ladder area codes contain {int(code_series.isna().sum())} "
            "missing value(s)."
        )
    stripped = code_series.astype(str).str.strip()
    blank = stripped == ""
    if blank.any():
        raise ValueError(
            f"ladder area codes contain {int(blank.sum())} blank value(s)."
        )
    # Group on stripped codes so a padded variant cannot split one area in
    # two, silently undercounting both halves.
    frame = pd.DataFrame({"code": stripped, "households": households})
    targets = frame.groupby("code", sort=True)["households"].sum().reset_index()
    summed = targets["households"].to_numpy(dtype=np.float64)
    if not np.isfinite(summed).all():
        raise ValueError("aggregated household targets must be finite.")
    zero = summed <= 0
    if zero.any():
        examples = targets.loc[zero, "code"].tolist()[:5]
        raise ValueError(
            "ladder-derived household targets must be positive; zero-count "
            f"area(s): {examples}."
        )
    return targets
