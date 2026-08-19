"""US capital gain distributions: split the Schedule D route out of LTCG.

The IRS reports capital gain distributions (RIC and REIT capital gain
dividends) through two routes: directly on Form 1040 line 7 for filers with
no other Schedule D activity (PUF ``E01100`` — already a separate PUF-stage
output, ``non_sch_d_capital_gains``) and on Schedule D line 13 for everyone
else, where they fold into net long-term gains and are no longer separable
in any microdata. The SOI Sales of Capital Assets study (latest: TY2015)
publishes the only all-route national total; no by-AGI CGD-specific series
exists anywhere in published SOI data. See
``microcosm/build/us/soca_capital_gain_distribution_shares.json`` for the
anchor arithmetic, the by-AGI tables that do exist, and the caveats.

The v1 split is therefore deterministic and proportional: eligible records
(positive long-term gains and no direct-route distributions — the two
routes are mutually exclusive on a real return) receive the national
Schedule-D share of long-term net gains. Participation margins and AGI-band
tilt are documented follow-ups in the resource file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_runtime import (
    SourceOperationSpec,
    SourceRuntimeContext,
    SourceRuntimeError,
)

__all__ = [
    "CapitalGainDistributionShares",
    "capital_gain_distribution_shares_asset_identity",
    "load_capital_gain_distribution_shares",
    "split_us_component_by_share_from_manifest",
]

_SPLIT_COMPONENT_PARAMETER_KEYS = frozenset(
    {
        "resource",
        "share_field",
        "source_column",
        "output",
        "exclusive_with",
    }
)

#: The only share resource the v1 handler knows how to read.
_SHARES_RESOURCE_NAME = "soca_capital_gain_distribution_shares"

#: Fields of the resource the manifest may select as the split share.
_ALLOWED_SHARE_FIELDS = frozenset({"schedule_d_cgd_share_of_lt_net_gains"})

#: Relative tolerance for re-deriving the packaged anchor arithmetic.
_ANCHOR_RTOL = 1e-9


@dataclass(frozen=True)
class CapitalGainDistributionShares:
    """Validated split parameters from the packaged SOCA-derived resource."""

    schedule_d_cgd_share_of_lt_net_gains: float
    anchor: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CapitalGainDistributionShares:
        anchor = raw.get("national_anchor_ty2015")
        if not isinstance(anchor, Mapping):
            raise ValueError(
                "capital gain distribution shares resource requires a "
                "'national_anchor_ty2015' mapping."
            )
        required = (
            "soca_all_route_capital_gain_distributions_k",
            "soca_long_term_total_net_gain_k",
            "pub1304_direct_1040_route_k",
            "schedule_d_route_residual_k",
            "schedule_d_cgd_share_of_lt_net_gains",
        )
        missing = [key for key in required if key not in anchor]
        if missing:
            raise ValueError(f"capital gain distribution anchor is missing {missing}.")
        all_route = float(anchor["soca_all_route_capital_gain_distributions_k"])
        lt_total = float(anchor["soca_long_term_total_net_gain_k"])
        direct = float(anchor["pub1304_direct_1040_route_k"])
        residual = float(anchor["schedule_d_route_residual_k"])
        share = float(anchor["schedule_d_cgd_share_of_lt_net_gains"])
        derived_residual = all_route - direct
        derived_share = derived_residual / (lt_total - direct)
        if not np.isclose(residual, derived_residual, rtol=_ANCHOR_RTOL):
            raise ValueError(
                "capital gain distribution anchor residual "
                f"{residual!r} does not equal all-route minus direct "
                f"({derived_residual!r}); the packaged numbers drifted."
            )
        if not np.isclose(share, derived_share, rtol=1e-6):
            raise ValueError(
                f"capital gain distribution anchor share {share!r} does not "
                f"re-derive from its own inputs ({derived_share!r})."
            )
        if not 0.0 < share < 1.0:
            raise ValueError(
                f"schedule_d_cgd_share_of_lt_net_gains must be in (0, 1), "
                f"got {share!r}."
            )
        return cls(
            schedule_d_cgd_share_of_lt_net_gains=share,
            anchor=dict(anchor),
        )


def load_capital_gain_distribution_shares() -> CapitalGainDistributionShares:
    """Load and validate the packaged SOCA-derived share resource."""

    path = files("microcosm.build.us") / f"{_SHARES_RESOURCE_NAME}.json"
    return _capital_gain_distribution_shares_from_bytes(path.read_bytes())


def _capital_gain_distribution_shares_from_bytes(
    payload: bytes,
) -> CapitalGainDistributionShares:
    """Resolve validated share semantics from the exact bytes being bound."""

    return CapitalGainDistributionShares.from_dict(json.loads(payload.decode("utf-8")))


def capital_gain_distribution_shares_asset_identity() -> dict[str, object]:
    """Bind the exact packaged bytes and resolved Schedule-D share semantics."""

    path = files("microcosm.build.us") / f"{_SHARES_RESOURCE_NAME}.json"
    payload = path.read_bytes()
    resolved = _capital_gain_distribution_shares_from_bytes(payload)
    return {
        "asset": f"microcosm.build.us/{_SHARES_RESOURCE_NAME}.json",
        "asset_sha256": hashlib.sha256(payload).hexdigest(),
        "schedule_d_cgd_share_of_lt_net_gains": (
            resolved.schedule_d_cgd_share_of_lt_net_gains
        ),
        "national_anchor_ty2015": dict(resolved.anchor),
    }


def split_us_component_by_share_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Split a memo component out of a source column at a declared share.

    Eligibility is structural: the source column must be positive and every
    ``exclusive_with`` column must be non-positive (the reporting routes are
    mutually exclusive on a real return). Ineligible records get zero. The
    split is deterministic — the declared share times the source column —
    so the stage needs no seed.
    """

    if operation.kind != "split_component_by_share":
        raise SourceRuntimeError(
            f"component split received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "split_component_by_share must follow read_table; there is no "
            "frame to split."
        )
    params = operation.parameters
    unexpected = sorted(set(params) - _SPLIT_COMPONENT_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"component split got unexpected parameter(s) {unexpected}; "
            f"allowed keys are {sorted(_SPLIT_COMPONENT_PARAMETER_KEYS)}."
        )
    resource = params.get("resource")
    if resource != _SHARES_RESOURCE_NAME:
        raise SourceRuntimeError(
            f"component split knows only the {_SHARES_RESOURCE_NAME!r} "
            f"resource, got {resource!r}."
        )
    share_field = params.get("share_field")
    if share_field not in _ALLOWED_SHARE_FIELDS:
        raise SourceRuntimeError(
            f"component split share_field must be one of "
            f"{sorted(_ALLOWED_SHARE_FIELDS)}, got {share_field!r}."
        )
    source_column = params.get("source_column")
    output = params.get("output")
    exclusive_with = params.get("exclusive_with", [])
    if not isinstance(source_column, str) or not source_column:
        raise SourceRuntimeError(
            "component split requires a non-empty string 'source_column'."
        )
    if not isinstance(output, str) or not output:
        raise SourceRuntimeError(
            "component split requires a non-empty string 'output'."
        )
    if not isinstance(exclusive_with, (list, tuple)) or not all(
        isinstance(name, str) and name for name in exclusive_with
    ):
        raise SourceRuntimeError(
            "component split 'exclusive_with' must be a list of column names."
        )
    if source_column not in frame.columns:
        raise SourceRuntimeError(
            f"component split source column {source_column!r} is not in the "
            "frame; the stage must run after the stage that produces it."
        )
    if output in frame.columns:
        raise SourceRuntimeError(
            f"component split output {output!r} already exists in the frame; "
            "refusing to overwrite."
        )
    missing_exclusive = [name for name in exclusive_with if name not in frame.columns]
    if missing_exclusive:
        raise SourceRuntimeError(
            f"component split exclusive_with column(s) {missing_exclusive} "
            "are not in the frame."
        )

    shares = load_capital_gain_distribution_shares()
    share = getattr(shares, share_field)

    values = pd.to_numeric(frame[source_column], errors="raise").astype(float)
    eligible = values > 0
    for name in exclusive_with:
        other = pd.to_numeric(frame[name], errors="raise").astype(float)
        eligible &= ~(other > 0)

    result = frame.copy()
    result[output] = np.where(eligible, values * share, 0.0)
    return result
