"""Weighted hot-deck draws of complete retirement contribution vectors.

Participation and amounts always travel together. A systematic weighted
quantile match approximates weighted frequencies within age/earnings cells;
it never draws account indicators or positive amounts independently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.build.source_runtime import SourceRuntimeError

RETIREMENT_COLUMNS = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)


def draw_joint_retirement_vectors(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    donor_weight: str,
    recipient_weight: str,
    seed: int,
    age_breaks: Sequence[float],
    positive_only: bool,
) -> pd.DataFrame:
    """Draw one whole donor vector, conditional on age and earnings sources.

    Both tables carry ``age``, ``has_wages``, ``has_self_employment``,
    ``rank_value`` and a unique ``donor_id``/``person_id``. Rank values are
    bundled contributions for CPS allocation and earned income for PUF
    imputation. Within a cell, weighted cumulative midpoints of recipients
    select the correspondingly ranked donor's entire vector. Equal ranks use
    seeded tie ordering after sorting identities, making row order irrelevant.

    Sparse cells widen age first, then earnings-source matching, but always
    exclude a donor with employee deferrals when the recipient has no wages,
    or SE-pension contributions when the recipient has no SE income. We do
    not erase one leg after drawing a joint vector. Zero-weight donors never
    participate. Positive CPS totals draw only positive donor vectors; PUF
    draws also include the observed nonsavers.
    """
    breaks = np.asarray(age_breaks, dtype=np.float64)
    if not np.isfinite(breaks).all() or np.any(np.diff(breaks) <= 0):
        raise SourceRuntimeError("Retirement age breaks must be finite and increasing.")
    features = ["age", "has_wages", "has_self_employment", "rank_value"]
    for table, identity, weight, targets in (
        (donor, "donor_id", donor_weight, list(RETIREMENT_COLUMNS)),
        (recipient, "person_id", recipient_weight, []),
    ):
        missing = sorted(set([identity, weight, *features, *targets]) - set(table))
        if missing:
            raise SourceRuntimeError(
                f"Joint retirement table missing columns: {missing}."
            )
        if table[identity].isna().any() or table[identity].duplicated().any():
            raise SourceRuntimeError(
                f"Joint retirement {identity} must be unique and nonmissing."
            )
        values = table[[weight, *features, *targets]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise SourceRuntimeError(
                "Joint retirement table contains nonfinite values."
            )
        if (table[[weight, *targets]].to_numpy(dtype=np.float64) < 0).any():
            raise SourceRuntimeError(
                "Joint retirement weights and contributions must be nonnegative."
            )
        for flag in ("has_wages", "has_self_employment"):
            if not table[flag].isin([False, True, 0, 1]).all():
                raise SourceRuntimeError(f"Joint retirement {flag} must be boolean.")

    source = donor.sort_values("donor_id").reset_index(drop=True)
    source = source.loc[source[donor_weight] > 0].reset_index(drop=True)
    if positive_only:
        source = source.loc[
            source[list(RETIREMENT_COLUMNS)].sum(axis=1) > 0
        ].reset_index(drop=True)
    if source.empty:
        raise SourceRuntimeError(
            "Joint retirement imputation has no positive-weight donors."
        )
    result = pd.DataFrame(0.0, index=recipient.index, columns=RETIREMENT_COLUMNS)
    source_age = np.searchsorted(breaks, source["age"].to_numpy(), side="right")
    target_age = np.searchsorted(breaks, recipient["age"].to_numpy(), side="right")
    source_wage = source["has_wages"].to_numpy(dtype=bool)
    source_se = source["has_self_employment"].to_numpy(dtype=bool)
    source_dc = source[list(RETIREMENT_COLUMNS[:2])].sum(axis=1).to_numpy() > 0
    source_pension = source[RETIREMENT_COLUMNS[4]].to_numpy() > 0
    cells = pd.DataFrame(
        {
            "age_cell": target_age,
            "wage": recipient["has_wages"].to_numpy(dtype=bool),
            "se": recipient["has_self_employment"].to_numpy(dtype=bool),
        }
    )
    rng = np.random.default_rng(seed)
    fallback_cells = 0
    cell_audit = []
    for (age_cell, wage, se), positions in cells.groupby(
        list(cells), sort=True
    ).indices.items():
        compatible = (wage | ~source_dc) & (se | ~source_pension)
        exact = compatible & (source_wage == wage) & (source_se == se)
        choices = [
            exact & (source_age == age_cell),
            exact,
            compatible & (source_age == age_cell),
            compatible,
        ]
        eligible = next((mask for mask in choices if mask.any()), None)
        if eligible is None:
            raise SourceRuntimeError(
                f"No compatible joint retirement donor for earnings cell {(wage, se)}."
            )
        fallback_cells += int(not choices[0].any())
        pool = source.loc[eligible].copy()
        targets = recipient.iloc[positions].sort_values("person_id").copy()
        # Stable rank sorts retain random ordering only within tied ranks.
        pool = pool.iloc[rng.permutation(len(pool))].sort_values(
            "rank_value", kind="stable"
        )
        targets = targets.iloc[rng.permutation(len(targets))].sort_values(
            "rank_value", kind="stable"
        )
        dw = pool[donor_weight].to_numpy(dtype=np.float64)
        cdf = np.cumsum(dw) / dw.sum()
        rw = targets[recipient_weight].to_numpy(dtype=np.float64)
        if rw.sum() == 0:
            rw = np.ones(len(targets))
        quantiles = (np.cumsum(rw) - rw / 2) / rw.sum()
        selected = np.searchsorted(cdf, quantiles, side="right").clip(0, len(pool) - 1)
        vectors = pool.iloc[selected][list(RETIREMENT_COLUMNS)].to_numpy()
        result.loc[targets.index] = vectors

        def shares(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
            """Summarize weighted participation in materialized NumPy vectors."""
            ira = values[:, 2:4].sum(axis=1) > 0
            plan = values[:, [0, 1, 4]].sum(axis=1) > 0
            return {
                name: float(weights[mask].sum() / weights.sum())
                for name, mask in {
                    "ira_only": ira & ~plan,
                    "employer_only": ~ira & plan,
                    "both": ira & plan,
                    "neither": ~ira & ~plan,
                }.items()
            }

        cell_audit.append(
            {
                "age_cell": int(age_cell),
                "has_wages": bool(wage),
                "has_self_employment": bool(se),
                "recipient_rows": len(targets),
                "donor_rows": len(pool),
                "fallback": not bool(choices[0].any()),
                "maximum_recipient_weight_share": float(rw.max() / rw.sum()),
                "source_joint_shares": shares(
                    pool[list(RETIREMENT_COLUMNS)].to_numpy(), dw
                ),
                "imputed_joint_shares": shares(vectors, rw),
            }
        )
    result.attrs["joint_retirement"] = {
        "method": "weighted_joint_hot_deck_v1",
        "fallback_cells": fallback_cells,
        "cells": cell_audit,
    }
    return result
