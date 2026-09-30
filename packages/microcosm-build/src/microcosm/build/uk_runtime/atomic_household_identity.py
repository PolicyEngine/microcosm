"""Explicit post-clone household keys for UK atomic assignment.

The host supplies original-source identity and ordered structural branches.
This helper neither infers ancestry from financial values/ID offsets nor
authenticates the supplied lineage. Source/EXPAND receipts remain authoritative.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from microcosm.graph.canonical import canonical_json

_BRANCHES = (
    "spi_support_channel",
    "cgt_incidence_clone",
    "cgt_band_donors",
    "geographic_support",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("UK geography identity: " + message)


def household_draw_key(
    *,
    source: str,
    source_vintage: str,
    source_household_id: int,
    clone_path: tuple[tuple[str, int], ...],
) -> str:
    """Encode an original-source ID and explicit ordered clone branch path.

    Every full-spine household needs a distinct key. Branch ordinals are
    supplied lineage identities, never sample row counters. Original branches
    should use their declared ordinal consistently; growing a pool must not
    change an existing path. Calibration K, weights, income, and final offset
    IDs are deliberately not inputs. Exact IDs above 2**53 remain distinct.
    """
    _require(
        all(
            type(value) is str
            and value
            and value.strip() == value
            and len(value) <= 256
            for value in (source, source_vintage)
        ),
        "qualified source and vintage required",
    )
    _require(
        isinstance(source_household_id, Integral)
        and not isinstance(source_household_id, (bool, np.bool_))
        and 0 < source_household_id < 2**63,
        "original source household ID must be an exact positive int64",
    )
    _require(type(clone_path) is tuple, "explicit immutable clone path required")
    path = []
    previous = -1
    for branch in clone_path:
        _require(
            type(branch) is tuple and len(branch) == 2,
            "named branch/ordinal pair required",
        )
        name, ordinal = branch
        _require(type(name) is str and name in _BRANCHES, "unknown structural branch")
        position = _BRANCHES.index(name)
        _require(position > previous, "clone branches must be unique and ordered")
        _require(
            isinstance(ordinal, Integral)
            and not isinstance(ordinal, (bool, np.bool_))
            and 0 <= ordinal < 2**32,
            "clone ordinal must be a bounded exact integer",
        )
        path.append([name, str(int(ordinal))])
        previous = position
    return (
        "uk-household-v1:"
        + canonical_json(
            [source, source_vintage, str(int(source_household_id)), path]
        ).decode()
    )
