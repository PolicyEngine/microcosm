"""Local target values derived from the sha-pinned UK OA ladder (#495).

The ladder artifact already carries census occupied-household counts per
output area with per-layer, per-country sha-pinned provenance — so household
count targets at any ladder grain need no new external pinning: they are the
artifact's own sums. This is the first bound local target family, and it is
universe-compatible with the FRS instrument (census occupied households vs
the survey's own household frame), unlike person-grain families that carry
the population_universe_private_households adjudication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.geography_ladder import UkOaLadder

__all__ = [
    "constituency_household_targets",
    "ladder_target_provenance",
    "local_authority_household_targets",
]


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
