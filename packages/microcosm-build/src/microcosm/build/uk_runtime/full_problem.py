"""Compile selected geographic contribution rows in the full UK build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import (
    UKRowwiseLocalMatrix,
    uk_local_target_surface,
)
from microcosm.build.uk_runtime.ledger_targets import _spec_geography
from microcosm.build.uk_runtime.local_rowwise import (
    build_uk_rowwise_local_surface_matrix,
    empty_uk_local_problem,
)
from microcosm.calibrate import TargetRegistry


def _national_contract_target_ids(registry: TargetRegistry) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(spec.metadata.get("contract_target_id", spec.name))
                for spec in registry.specs
                if _spec_geography(spec)[0] == "country"
            }
        )
    )


def _joint_surface_registry(
    local_registry: TargetRegistry,
    national_registry: TargetRegistry,
) -> TargetRegistry:
    """Put country controls beside local cells for declared reconciliation.

    Regional constraints stay in the national solve registry. They are outside
    the country/constituency/LA reconciliation rule and cannot be passed as
    country controls or silently assigned a new reconciliation policy.
    """

    return TargetRegistry(
        [
            *local_registry.specs,
            *(
                spec
                for spec in national_registry.specs
                if _spec_geography(spec)[0] == "country"
            ),
        ],
        country="uk",
    )


def build_uk_full_local_problem(
    assignment: Any,
    *,
    local_registry: TargetRegistry,
    national_registry: TargetRegistry,
    local_metrics: Mapping[str, pd.DataFrame],
    period: int,
    sample_fraction: float,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]],
    census_household_uprating: Mapping[str, Any] | None = None,
    selected_surface: pd.DataFrame | None = None,
    surface_receipt: Mapping[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    UKRowwiseLocalMatrix,
    dict[str, Any],
    tuple[str, ...],
    dict[str, Any],
]:
    household = assignment.result.frame.table("household").reset_index(drop=True)
    household_index = pd.Index(household["household_id"], name="household_id")
    metrics = {
        grain: frame.set_axis(household_index, axis="index")
        for grain, frame in local_metrics.items()
    }
    assigned = {
        "constituency": pd.Series(
            household["constituency_code"].astype(str).to_numpy(),
            index=household_index,
        ),
        "la": pd.Series(
            household["local_authority_code"].astype(str).to_numpy(),
            index=household_index,
        ),
    }
    assigned = {grain: assigned[grain] for grain in metrics}
    national_ids = _national_contract_target_ids(national_registry)
    if selected_surface is None:
        surface, cross_grain = uk_local_target_surface(
            _joint_surface_registry(local_registry, national_registry),
            bound_national_target_ids=national_ids,
            period=period,
            reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
            census_household_uprating=census_household_uprating,
        )
    else:
        surface = selected_surface.copy()
        cross_grain = dict(surface_receipt or {})
    covered = {
        grain: set(values.astype(str).tolist()) for grain, values in assigned.items()
    }
    covered_mask = pd.Series(
        [
            str(row.area_code) in covered[str(row.area_type)]
            for row in surface.itertuples(index=False)
        ],
        index=surface.index,
        dtype=bool,
    )
    dropped = surface.loc[~covered_mask]
    if sample_fraction < 1.0:
        surface = surface.loc[covered_mask].reset_index(drop=True)
    # Below f100 a covered area can still carry a nonzero cell with no metric
    # support in the sample (no self-employed household among three drawn
    # rows). The builder refuses such a cell at every rung; at development
    # rungs the cell is dropped here and receipted instead. f100 stays strict.
    unreachable = surface.iloc[0:0]
    if sample_fraction < 1.0 and len(surface):
        nonzero_by_grain = {
            grain: (metrics[grain] != 0).groupby(assigned[grain]).sum()
            for grain in metrics
        }
        unreachable_mask = pd.Series(
            [
                float(row.value) != 0.0
                and str(row.metric) in nonzero_by_grain[str(row.area_type)].columns
                and str(row.area_code) in nonzero_by_grain[str(row.area_type)].index
                and int(
                    nonzero_by_grain[str(row.area_type)].loc[
                        str(row.area_code), str(row.metric)
                    ]
                )
                == 0
                for row in surface.itertuples(index=False)
            ],
            index=surface.index,
            dtype=bool,
        )
        unreachable = surface.loc[unreachable_mask]
        surface = surface.loc[~unreachable_mask].reset_index(drop=True)
    rung_surface = {
        "dropped_unreachable_cells": int(len(unreachable)),
        "dropped_unreachable_by_grain": {
            str(key): int(value)
            for key, value in unreachable.groupby("area_type").size().items()
        },
        "dropped_unreachable_by_family": {
            str(key): int(value)
            for key, value in unreachable.groupby("family").size().items()
        },
        "fraction": float(sample_fraction),
        "dropped_cells": int(len(dropped) if sample_fraction < 1.0 else 0),
        "dropped_by_grain": (
            {
                str(key): int(value)
                for key, value in dropped.groupby("area_type").size().items()
            }
            if sample_fraction < 1.0
            else {}
        ),
        "dropped_by_family": (
            {
                str(key): int(value)
                for key, value in dropped.groupby("family").size().items()
            }
            if sample_fraction < 1.0
            else {}
        ),
    }
    rosters = {
        "constituency": tuple(map(str, np.unique(assignment.ladder.constituency_code))),
        "la": tuple(map(str, np.unique(assignment.ladder.local_authority_code))),
    }
    if surface.empty:
        problem = empty_uk_local_problem(household_index)
    else:
        problem = build_uk_rowwise_local_surface_matrix(
            metrics,
            assigned,
            surface,
            area_codes_by_grain={grain: rosters[grain] for grain in metrics},
            require_every_assigned_area_covered=(sample_fraction == 1.0),
        )
    local_bound = tuple(
        sorted(
            {
                f"{row.family}/{row.area_type}"
                for row in surface[["family", "area_type"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
        )
    )
    national_bound = tuple(
        f"national/{family}"
        for family in sorted({spec.family for spec in national_registry.specs})
    )
    return (
        household,
        problem,
        cross_grain,
        (*local_bound, *national_bound),
        rung_surface,
    )
